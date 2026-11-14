import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from mixxx_cue_guess import baseline
from mixxx_harness.testing import make_db

SR, BPM, START = 22050, 128.0, 0.3
BEAT = 60 / BPM


def bar(n: int) -> float:
    return START + 4 * n * BEAT


def synth(path: Path, bars: int = 64) -> None:
    """Kicks on every beat; bars 16-47 are louder with noise on top, so the changes are at 16 and 48."""
    rng = np.random.default_rng(0)
    x = np.zeros(int(SR * (bar(bars) + 1)), dtype=np.float32)
    t = np.arange(int(0.15 * SR)) / SR
    kick = (np.sin(2 * np.pi * 55 * t) * np.exp(-t * 30)).astype(np.float32)
    for b in range(bars * 4):
        i = int((START + b * BEAT) * SR)
        x[i:i + len(kick)] += kick * (0.9 if 64 <= b < 192 else 0.3)
    loud = slice(int(bar(16) * SR), int(bar(48) * SR))
    x[loud] += 0.1 * rng.standard_normal(loud.stop - loud.start).astype(np.float32)
    sf.write(path, np.stack([x, x], axis=1), SR)


class Matching(unittest.TestCase):
    def test_each_guess_and_cue_once(self):
        self.assertEqual(baseline.matched([10.0, 10.1], [10.05], 0.5), 1)
        self.assertEqual(baseline.matched([10.0, 20.0], [10.2, 20.6, 30.0], 0.5), 1)
        self.assertEqual(baseline.matched([], [1.0], 0.5), 0)


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.wav = cls.tmp / "music" / "track.wav"
        cls.wav.parent.mkdir()
        synth(cls.wav)

    def test_finds_the_changes(self):
        total, bass, fps = baseline.envelopes(str(self.wav))
        g = baseline.guess(total, bass, fps, BPM)
        self.assertLess(abs(g["beats"][0] - START), 0.05)
        for want in (bar(16), bar(48)):
            self.assertLess(min(abs(t - want) for t in g["start + top 3"]), BEAT / 2)

    def test_end_to_end_on_a_test_db(self):
        dbfile = self.tmp / "mixxxdb.sqlite"
        make_db(dbfile, [(str(self.wav), 4), (str(self.tmp / "music/unmarked.wav"), 2)], str(self.wav.parent))
        con = sqlite3.connect(dbfile)
        con.execute("UPDATE library SET bpm = ?, samplerate = ?, channels = 2 WHERE id = 1", (BPM, SR))
        for h, t in enumerate([START, bar(16), bar(32), bar(48)]):
            con.execute("UPDATE cues SET position = ? WHERE track_id = 1 AND hotcue = ?", (t * SR * 2, h))
        con.commit()
        con.close()
        out = io.StringIO()
        cfg = mock.Mock(mixxx_db=self.tmp / "not-this.sqlite")
        with mock.patch.object(baseline.host, "load_config", return_value=cfg), contextlib.redirect_stdout(out):
            self.assertEqual(baseline.main(["--db", str(dbfile), "-j", "1"]), 0)
        text = out.getvalue()
        self.assertIn("1 marked tracks", text)
        self.assertIn("within half a beat of the beat grid: 100%", text)
        self.assertRegex(text, r"start \+ top 3\s+(75|100)%")

    def test_refuses_the_live_db(self):
        cfg = mock.Mock(mixxx_db=self.wav)
        with mock.patch.object(baseline.host, "load_config", return_value=cfg), self.assertRaises(SystemExit):
            baseline.main(["--db", str(self.wav)])


if __name__ == "__main__":
    unittest.main()
