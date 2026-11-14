"""Guess 4 hot cues for each track that has none, for the mod to write at the next launch.

    python -m mixxx_cue_guess.guess              # the harness's newest backup
    python -m mixxx_cue_guess.guess --limit 25 -j 3

Reads a copy of mixxxdb.sqlite and your audio; writes only <state_dir>/cue-guess/guesses.json.
Trains the ranker on every marked track (about 2 minutes per 400 tracks), then guesses each
track with no hot cues and a BPM that the mod hasn't written to before. Guesses are the
ranker's 2 drops, each with a cue 8 bars before it, as hot cues 1-4 in time order (your pads
always run in time order: 442 of 442 marked tracks, 2026-11-14).
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mixxx_cue_guess import baseline as b, ranker, read_json, state_folder, write_json
from mixxx_harness import db, host


def targets(con, written: set[int]) -> list[dict]:
    """Tracks with no hot cues and a BPM, not written to before."""
    rows = con.execute("""SELECT l.id, l.bpm FROM library l WHERE l.mixxx_deleted = 0 AND l.bpm > 0
                          AND NOT EXISTS (SELECT 1 FROM cues c WHERE c.track_id = l.id AND c.hotcue >= 0)""")
    return [{"id": tid, "bpm": bpm, "path": db.track_path(con, tid)} for tid, bpm in rows if tid not in written]


def analyze(track: dict) -> dict:
    """Candidates and features for one unmarked track, or {"skip": reason}. Runs in a worker."""
    if not track["path"] or not Path(track["path"]).is_file():
        return {"skip": "file not found"}
    try:
        total, bass, fps = b.envelopes(track["path"])
        mel, chroma = ranker.spectra(track["path"])
    except Exception as e:  # e.g. a format libsndfile can't read
        return {"skip": f"can't decode ({Path(track['path']).suffix or 'no extension'})", "error": str(e)}
    beats, idx, X = ranker.candidates(total, bass, fps, track["bpm"], mel, chroma)
    return {"beats": beats / fps, "idx": idx, "X": X}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="a copy of mixxxdb.sqlite (default: the harness's newest backup)")
    ap.add_argument("--limit", type=int, help="guess at most N tracks this run")
    ap.add_argument("-j", "--jobs", type=int, default=3, help="worker processes (default 3: each holds a track's audio)")
    args = ap.parse_args(argv)
    cfg = host.load_config()
    path = Path(args.db).expanduser() if args.db else b.newest_backup(cfg)
    if path.resolve() == cfg.mixxx_db.resolve():
        sys.exit("That's Mixxx's live database; pass a copy (the harness's backups are copies)")
    folder = state_folder()
    written = set(read_json(folder / "written.json", []))
    guesses = read_json(folder / "guesses.json", {})
    con = db.connect(path)
    try:
        marked = b.marked_tracks(con)
        todo = [t for t in targets(con, written) if str(t["id"]) not in guesses][: args.limit]
    finally:
        con.close()
    print(f"Database: {path}\nTraining on {len(marked)} marked tracks, then guessing {len(todo)} tracks")
    with ProcessPoolExecutor(args.jobs) as pool:
        train = [r for r in pool.map(ranker.prepare, marked, chunksize=4) if "skip" not in r and len(r["y"])]
        if not train:
            sys.exit("No marked track could be analyzed, so there's nothing to learn from")
        model = ranker.fit(np.vstack([t["X"] for t in train]), np.concatenate([t["y"] for t in train]))
        results = list(pool.map(analyze, todo, chunksize=2))

    skips = Counter()
    for t, r in zip(todo, results):
        if "skip" in r:
            skips[r["skip"]] += 1
            continue
        seconds = ranker.pick(r, ranker.score(model, r["X"])) if len(r["idx"]) else []
        if len(set(seconds)) != 4:  # fewer than 2 drops, or a drop in the first 8 bars
            skips["no 2 drops found"] += 1
            continue
        guesses[str(t["id"])] = {"path": t["path"], "seconds": [round(s, 4) for s in seconds]}
    write_json(folder / "guesses.json", guesses)
    done = len(todo) - sum(skips.values())
    print(f"Guessed {done}, skipped {sum(skips.values())}" + "".join(f"\n  {n} {why}" for why, n in skips.most_common()))
    print(f"{len(guesses)} guesses wait in {folder / 'guesses.json'}; the next harness launch writes them into Mixxx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
