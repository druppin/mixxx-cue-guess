"""The mixxx-harness mod: writes guessed hot cues into Mixxx for you to review.

1. `python -m mixxx_cue_guess.guess` (slow: it reads audio) trains the ranker on your marked
   tracks and guesses 4 hot cues for each track with none. It writes them to
   <state_dir>/cue-guess/guesses.json and touches no database.
2. At each harness launch (Mixxx closed), this mod writes those guesses as hot cues, colored
   GUESS_COLOR and labelled GUESS_LABEL, and adds each track to REVIEW_CRATE. cratesync doesn't
   count a track in that crate as marked.
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
        counts = db.hot_cue_counts(con)
        new = []
        for key, g in guesses.items():
            tid = int(key)
            if tid in written or counts.get(tid, 1) or db.track_path(con, tid) != g["path"]:
                continue  # done before, has cues now (or is gone), or the row points at another file
            rate, channels = db.audio_format(con, tid)
            if rate and channels:  # positions count samples across all channels (cratesync findings, spike 1)
                new.append((tid, [(s * rate * channels, GUESS_COLOR, GUESS_LABEL) for s in g["seconds"]]))
        if not approved and not new:
            return
        with ctx.write():
            for cue in approved:
                ctx.set_cue_look(cue, APPROVED_COLOR, "")
            for tid, cues in new:
                ctx.add_hot_cues(tid, cues)
                ctx.add_to_crate(REVIEW_CRATE, tid)
        if not ctx.dry_run:
            write_json(folder / "written.json", sorted(written | {tid for tid, _ in new}))


mod = CueGuess()
