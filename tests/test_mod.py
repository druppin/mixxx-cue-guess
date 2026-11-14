import contextlib
import io
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mixxx_cue_guess import APPROVED_COLOR, GUESS_COLOR, GUESS_LABEL, REVIEW_CRATE, CueGuess, guess
from mixxx_harness import db, mods
from mixxx_harness.testing import make_db
from test_baseline import BPM, SR, bar, synth

RATE, CH = 48000, 2


class Mod(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "mixxxdb.sqlite"
        self.paths = [str(self.tmp / f"{n}.mp3") for n in "abc"]
        make_db(self.db, [(self.paths[0], 4), (self.paths[1], 0), (self.paths[2], 0)], str(self.tmp))
        con = sqlite3.connect(self.db)
        con.execute("UPDATE library SET samplerate = ?, channels = ?", (RATE, CH))
        con.commit()
        con.close()
        self.state = self.tmp / "state"
        self.write_guesses({2: [10.0, 20.0, 60.0, 70.0], 1: [1.0, 2.0, 3.0, 4.0],
                            3: [5.0, 6.0, 7.0, 8.0]}, paths={3: "/elsewhere/c.mp3"})
        patcher = mock.patch.object(db, "mixxx_running", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_guesses(self, by_id, paths=None):
        self.state.mkdir(exist_ok=True)
        data = {str(t): {"path": (paths or {}).get(t, self.paths[t - 1]), "seconds": s} for t, s in by_id.items()}
        (self.state / "guesses.json").write_text(json.dumps(data))

    def run_mod(self, dry_run=False):
        return mods.run(self.db, [CueGuess(self.state)], self.tmp / "bk", self.tmp / "actions.log",
                        dry_run=dry_run, when="before")

    def cues(self, tid):
        con = db.connect(self.db)
        return [tuple(r) for r in con.execute("SELECT hotcue, type, position, length, color, label FROM cues "
                                              "WHERE track_id = ? AND hotcue >= 0 ORDER BY hotcue", (tid,))]

    def test_writes_guesses_only_onto_tracks_without_cues_at_the_same_path(self):
        done, warnings = self.run_mod()
        self.assertEqual(warnings, [])
        s = RATE * CH
        self.assertEqual(self.cues(2), [(h, 1, p * s, 0, GUESS_COLOR, GUESS_LABEL) for h, p in enumerate([10, 20, 60, 70])])
        self.assertEqual(len(self.cues(1)), 4)  # had cues: untouched
        self.assertEqual(self.cues(3), [])  # the row points at another file now
        self.assertEqual(list(db.crate_members(db.connect(self.db), REVIEW_CRATE)), [2])
        self.assertEqual(json.loads((self.state / "written.json").read_text()), [2])

    def test_approval_recolors_and_a_cleared_track_is_not_guessed_again(self):
        self.run_mod()
        self.run_mod()  # still in the crate: nothing changes
        self.assertEqual({c[4] for c in self.cues(2)}, {GUESS_COLOR})
        con = db.connect(self.db, write=True)
        db.remove_from_crate(con, REVIEW_CRATE, 2)  # you approve it in Mixxx
        con.commit()
        self.run_mod()
        self.assertEqual({(c[4], c[5]) for c in self.cues(2)}, {(APPROVED_COLOR, "")})
        con = db.connect(self.db, write=True)
        con.execute("DELETE FROM cues WHERE track_id = 2")  # you clear it instead
        con.commit()
        self.run_mod()
        self.assertEqual(self.cues(2), [])

    def test_status_counts_each_stage(self):
        from mixxx_cue_guess import status
        con = db.connect(self.db)
        st = status(con, self.state)
        self.assertEqual((st["waiting"], st["in review"], st["done (left the crate)"]), (1, 0, 0))  # track 2 only
        self.run_mod()
        st = status(db.connect(self.db), self.state)
        self.assertEqual((st["waiting"], st["in review"], st["review list"][0][0]), (0, 1, 2))
        con = db.connect(self.db, write=True)
        db.remove_from_crate(con, REVIEW_CRATE, 2)
        con.commit()
        st = status(db.connect(self.db), self.state)
        self.assertEqual((st["in review"], st["approved, not recolored yet"], st["done (left the crate)"]), (0, 1, 1))
        self.assertEqual(st["need analysis in Mixxx first"], 1)  # track 3: no cues, no BPM in the test db

    def markers(self, tid):
        q = "SELECT type, position, length FROM cues WHERE track_id = ? AND type IN (6, 7) ORDER BY type"
        return [tuple(r) for r in db.connect(self.db).execute(q, (tid,))]

    def test_intro_and_outro_only_where_unset_and_only_in_review(self):
        s, end = RATE * CH, 100.0 * RATE * CH
        con = sqlite3.connect(self.db)
        for tid, intro_len, outro_pos in ((1, 700, 50 * s), (2, 0, -1)):  # track 1: yours; track 2: unset
            con.execute("INSERT INTO cues (track_id, type, position, length, hotcue) VALUES (?, 6, 1000, ?, -1)", (tid, intro_len))
            con.execute("INSERT INTO cues (track_id, type, position, length, hotcue) VALUES (?, 7, ?, ?, -1)",
                        (tid, outro_pos, end if outro_pos == -1 else end - outro_pos))
        con.commit()
        con.close()
        data = json.loads((self.state / "guesses.json").read_text())
        for t in ("1", "2"):
            data[t].update({"intro": 5.0, "outro": 80.0})
        (self.state / "guesses.json").write_text(json.dumps(data))
        mine = self.markers(1)
        done, warnings = self.run_mod()
        self.assertEqual(warnings, [])
        self.assertEqual(self.markers(2), [(6, 1000, 5 * s - 1000), (7, 80 * s, end - 80 * s)])
        self.assertEqual(self.markers(1), mine)  # yours: untouched

        con = sqlite3.connect(self.db)  # a track you approved before its markers were guessed: left alone
        con.execute("UPDATE cues SET length = 0 WHERE track_id = 2 AND type = 6")
        con.commit()
        con.close()
        c = db.connect(self.db, write=True)
        db.remove_from_crate(c, REVIEW_CRATE, 2)
        c.commit()
        self.run_mod()
        self.assertEqual(self.markers(2)[0], (6, 1000, 0))

    def test_unapproved_guesses_are_not_training_data(self):
        from mixxx_cue_guess import baseline
        self.run_mod()  # track 2 gets 4 guessed cues
        self.assertEqual([t["id"] for t in baseline.marked_tracks(db.connect(self.db))], [1])
        c = db.connect(self.db, write=True)
        db.remove_from_crate(c, REVIEW_CRATE, 2)
        c.commit()
        self.run_mod()  # approved: labels cleared, so it's yours now
        self.assertEqual([t["id"] for t in baseline.marked_tracks(db.connect(self.db))], [1, 2])

    def test_dry_run_writes_nothing(self):
        before = self.db.read_bytes()
        done, warnings = self.run_mod(dry_run=True)
        self.assertEqual(warnings, [])
        self.assertTrue(any("add 4 hot cues" in line for line in done["cue-guess"]))
        self.assertEqual(self.db.read_bytes(), before)
        self.assertFalse((self.state / "written.json").exists())


class GuessCommand(unittest.TestCase):
    def test_guesses_unmarked_tracks_from_marked_ones(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "music").mkdir()
        wavs = [tmp / "music" / f"{n}.wav" for n in range(4)]
        synth(wavs[0])
        for w in wavs[1:]:
            shutil.copy(wavs[0], w)
        dbfile = tmp / "mixxxdb.sqlite"
        make_db(dbfile, [(str(w), 4 if i < 3 else 0) for i, w in enumerate(wavs)], str(tmp / "music"))
        con = sqlite3.connect(dbfile)
        con.execute("UPDATE library SET bpm = ?, samplerate = ?, channels = 2", (BPM, SR))
        for h, t in enumerate([bar(8), bar(16), bar(32), bar(40)]):  # the synth's one rise is at bar 16
            con.execute("UPDATE cues SET position = ? WHERE track_id < 4 AND hotcue = ?", (t * SR * 2, h))
        for tid in (1, 2, 3):  # your intro ends on your first cue; your outro starts 8 bars after your last
            con.execute("INSERT INTO cues (track_id, type, position, length, hotcue) VALUES (?, 6, 0, ?, -1)", (tid, bar(8) * SR * 2))
            con.execute("INSERT INTO cues (track_id, type, position, length, hotcue) VALUES (?, 7, ?, ?, -1)",
                        (tid, bar(48) * SR * 2, (bar(64) - bar(48)) * SR * 2))
        con.commit()
        con.close()
        cfg = mock.Mock(mixxx_db=tmp / "not-this.sqlite")
        with mock.patch.object(guess.host, "load_config", return_value=cfg), \
                mock.patch.object(guess, "state_folder", return_value=tmp / "state"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(guess.main(["--db", str(dbfile), "-j", "1"]), 0)
        out = json.loads((tmp / "state" / "guesses.json").read_text())
        self.assertEqual(list(out), ["4"])
        seconds = out["4"]["seconds"]
        self.assertEqual(len(seconds), 4)
        self.assertEqual(seconds, sorted(seconds))
        self.assertTrue(any(abs(s - bar(16)) < 60 / BPM for s in seconds))
        self.assertLessEqual(out["4"]["intro"], seconds[0] + 60 / BPM)  # at or before the first cue
        self.assertGreater(out["4"]["outro"], seconds[-1])  # after the last

    def test_refuses_the_live_db(self):
        cfg = mock.Mock(mixxx_db=Path("/tmp/live.sqlite"))
        with mock.patch.object(guess.host, "load_config", return_value=cfg), self.assertRaises(SystemExit):
            guess.main(["--db", "/tmp/live.sqlite"])


if __name__ == "__main__":
    unittest.main()
