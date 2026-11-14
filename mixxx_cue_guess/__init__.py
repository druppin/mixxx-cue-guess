"""The mixxx-harness mod: writes guessed hot cues into Mixxx for you to review.

1. `python -m mixxx_cue_guess.guess` (slow: it reads audio) trains the ranker on your marked
   tracks and guesses 4 hot cues for each track with none. It writes them to
   <state_dir>/cue-guess/guesses.json and touches no database.
2. At each harness launch (Mixxx closed), this mod writes those guesses as hot cues, colored
   GUESS_COLOR and labelled GUESS_LABEL, and adds each track to REVIEW_CRATE. cratesync doesn't
   count a track in that crate as marked.
   It also sets the intro end and outro start where those aren't set yet, on tracks in REVIEW_CRATE.
3. You check the cues in Mixxx, fix any that are wrong, and remove the track from REVIEW_CRATE.
   That is the approval. At the next launch the mod turns that track's guessed cues
   APPROVED_COLOR and clears their label, so the guess color always means "not checked yet".

A track is guessed once: written.json lists every track the mod wrote cues to, so one you
cleared by hand isn't guessed again. A track that has any hot cues is never written to.
"""
import json
import os
from pathlib import Path

from mixxx_harness import db
from mixxx_harness.mods import Owns

REVIEW_CRATE = "Cues: review"  # cratesync's laptop/mixxx MARKED rule names it too
GUESS_COLOR = 0x0044FF  # blue; no hot cue in the library used it (2026-11-14)
GUESS_LABEL = "guess"
APPROVED_COLOR = 0xFF8000  # orange: 1797 of the library's 1809 hot cues (2026-11-14)


def state_folder() -> Path:
    from mixxx_harness import host
    return host.load_config().state_dir / "cue-guess"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write_json(path: Path, data) -> None:
    """Whole file or nothing: a crash mid-write leaves the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def writable(con, guesses: dict, written: set[int]) -> list[tuple[int, list[tuple[float, int, str]]]]:
    """The guesses the mod would write now, as (track id, [(position, color, label)])."""
    counts = db.hot_cue_counts(con)
    out = []
    for key, g in guesses.items():
        tid = int(key)
        if tid in written or counts.get(tid, 1) or db.track_path(con, tid) != g["path"]:
            continue  # done before, has cues now (or is gone), or the row points at another file
        rate, channels = db.audio_format(con, tid)
        if rate and channels:  # positions count samples across all channels (cratesync findings, spike 1)
            out.append((tid, [(s * rate * channels, GUESS_COLOR, GUESS_LABEL) for s in g["seconds"]]))
    return out


def marker_writes(con, guesses: dict, tracks: set[int]) -> list[tuple[int, str, float]]:
    """(track id, "intro" or "outro", position) for each guessed intro end and outro start the mod
    would write now: on `tracks` only, at the guessed file, and only where that marker isn't set.
    Layout: mixxx_cue_guess.markers."""
    out = []
    for key, g in guesses.items():
        tid = int(key)
        if tid not in tracks or db.track_path(con, tid) != g["path"]:
            continue
        rate, channels = db.audio_format(con, tid)
        if not (rate and channels):
            continue
        intro = con.execute("SELECT position, length FROM cues WHERE track_id = ? AND type = 6", (tid,)).fetchall()
        outro = con.execute("SELECT position, length FROM cues WHERE track_id = ? AND type = 7", (tid,)).fetchall()
        if g.get("intro") is not None and len(intro) == 1 and intro[0][1] == 0 and 0 <= intro[0][0] < g["intro"] * rate * channels:
            out.append((tid, "intro", g["intro"] * rate * channels))
        if g.get("outro") is not None and len(outro) == 1 and outro[0][0] == -1 and 0 <= g["outro"] * rate * channels < outro[0][1]:
            out.append((tid, "outro", g["outro"] * rate * channels))
    return out


def status(con, folder: Path) -> dict:
    """Where every track stands, for the dashboard page. Reads only."""
    guesses = read_json(folder / "guesses.json", {})
    written = set(read_json(folder / "written.json", []))
    counts = db.hot_cue_counts(con)
    review = db.crate_members(con, REVIEW_CRATE)
    bpm = dict(con.execute("SELECT id, bpm FROM library WHERE mixxx_deleted = 0"))
    uncued = [t for t, n in counts.items() if not n and t not in written]
    return {
        "waiting": len(writable(con, guesses, written)),
        "in review": len(review),
        "approved, not recolored yet": len({t for (t,) in con.execute(
            "SELECT DISTINCT track_id FROM cues WHERE hotcue >= 0 AND label = ?", (GUESS_LABEL,))} - set(review)),
        "done (left the crate)": len(written - set(review)),
        "to guess": sum(1 for t in uncued if (bpm.get(t) or 0) > 0 and str(t) not in guesses),
        "need analysis in Mixxx first": sum(1 for t in uncued if not (bpm.get(t) or 0) > 0),
        "review list": sorted(review.items()),
    }


class CueGuess:
    name, version = "cue-guess", "0.2.0"
    owns = Owns(crates=frozenset({REVIEW_CRATE}), new_cues=True)

    def __init__(self, folder: Path | None = None):
        self._folder = folder  # tests pass one; otherwise the harness's state folder

    def session(self, ctx) -> None:
        folder = self._folder or state_folder()
        guesses = read_json(folder / "guesses.json", {})
        written = set(read_json(folder / "written.json", []))
        con = ctx.con
        review = set(db.crate_members(con, REVIEW_CRATE))
        approved = [cue for cue, tid in con.execute(
            "SELECT id, track_id FROM cues WHERE hotcue >= 0 AND label = ?", (GUESS_LABEL,)) if tid not in review]
        new = writable(con, guesses, written)
        markers = marker_writes(con, guesses, review | {tid for tid, _ in new})  # never on a track you approved
        if not approved and not new and not markers:
            return
        with ctx.write():
            for cue in approved:
                ctx.set_cue_look(cue, APPROVED_COLOR, "")
            for tid, cues in new:
                ctx.add_hot_cues(tid, cues)
                ctx.add_to_crate(REVIEW_CRATE, tid)
            for tid, kind, pos in markers:
                (ctx.set_intro_end if kind == "intro" else ctx.set_outro_start)(tid, pos)
        if not ctx.dry_run:
            write_json(folder / "written.json", sorted(written | {tid for tid, _ in new}))

    def page(self):
        from mixxx_harness.host import load_config
        from mixxx_cue_guess.page import Page  # PySide6 only when the dashboard opens
        return Page(load_config().mixxx_db, self._folder or state_folder())


mod = CueGuess()
