"""Trained ranker: learns which rises in a track you cue as drops, scored on tracks it didn't train on.

    python -m mixxx_cue_guess.ranker              # the harness's newest backup
    python -m mixxx_cue_guess.ranker --db copy.sqlite

Reads only: a copy of mixxxdb.sqlite and your audio files. Writes nothing.

1. Candidates: every beat where the 1-bar rise peaks within 2 bars either side (the point
   the baseline's "2 drops" rule sharpens a drop to).
2. Labels: a candidate is a drop if it's within a beat of one of your cues that has another of
   your cues 8 bars before it (your A, A+8 bars pattern); on tracks without that pattern, any cue.
3. Model: gradient boosting on a few features per candidate (rise sizes, place in the track,
   loudness after vs. before, fit to the track's phrase grid).
4. Guess: the 2 top-scored candidates 16+ bars apart, each with a cue 8 bars before it.
Scored with 5-fold cross-validation by track, against the "2 drops" rule on the same tracks.
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mixxx_cue_guess import baseline as b
from mixxx_harness import db, host

FOLDS = 5
FEATURES = ("8-bar rise", "1-bar rise", "rise vs. track's biggest", "place in track", "bars from start",
            "loudness after", "bass after", "loudness before", "on the phrase grid")


def candidates(total, bass, fps: float, bpm: float):
    """Beat frames, candidate beat indices, and one feature row per candidate."""
    period = fps * 60 / bpm
    beats = b.beat_frames(bass, period)
    width = int(round(period * b.PHRASE))
    long = b.rise(total, bass, beats, width)
    short = b.rise(total, bass, beats, int(round(period * 4)))
    idx = np.array([i for i in range(len(beats)) if np.isfinite(long[i]) and b.sharpen(short, i) == i], dtype=int)
    if not len(idx):
        return beats, idx, np.zeros((0, len(FEATURES)))
    after = lambda v: b.window_mean(v, beats[idx], beats[idx] + width) - np.median(v)  # noqa: E731
    before = b.window_mean(total, beats[idx] - width, beats[idx]) - np.median(total)
    top = idx[np.argmax(long[idx])]
    off = (idx - top) % b.PHRASE
    X = np.column_stack([long[idx], short[idx], long[idx] / max(long[idx].max(), b.EPS), idx / len(beats),
                         idx / 4 / 100, after(total), after(bass), before, 1 - np.minimum(off, b.PHRASE - off) / 16])
    ok = np.isfinite(X).all(axis=1)  # windows past the track's end
    return beats, idx[ok], X[ok]


def drop_cues(cues, beat: float) -> list[float]:
    """Your cues with another cue 8 bars before them; all cues if none do."""
    found = [c for c in cues if any(abs(c - 32 * beat - o) <= beat for o in cues)]
    return found or list(cues)


def prepare(track: dict) -> dict:
    """Features, labels and the rule's guesses for one track, or {"skip": reason}. Runs in a worker."""
    loaded = b.load(track)
    if isinstance(loaded, dict):
        return loaded
    total, bass, fps = loaded
    beats, idx, X = candidates(total, bass, fps, track["bpm"])
    beat = 60 / track["bpm"]
    times = beats[idx] / fps
    labels = drop_cues(track["cues"], beat)
    y = np.array([any(abs(t - c) <= beat for c in labels) for t in times], dtype=float)
    rule = b.guess(total, bass, fps, track["bpm"])["2 drops"]
    return {"X": X, "y": y, "idx": idx, "beats": beats / fps, "cues": track["cues"], "beat": beat,
            "rule": rule, "reachable": sum(any(abs(t - c) <= beat for t in times) for c in labels), "drops": len(labels)}


def fit(X: np.ndarray, y: np.ndarray):
    """Gradient boosting: a few hundred small trees. Beat logistic regression by 5 points on the library (44% vs 49%)."""
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(X, y)


def score(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def pick(t: dict, s: np.ndarray) -> list[float]:
    """The 2 best-scored candidates 16+ bars apart, each with a cue 8 bars before it, in seconds."""
    r = np.full(len(t["beats"]), -np.inf)
    r[t["idx"]] = s
    found = b.drops(r, 2) if len(t["idx"]) else []
    found = [d for d in found if np.isfinite(r[d])]
    return sorted(float(t["beats"][i]) for i in found + [max(d - b.PHRASE, 0) for d in found])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="a copy of mixxxdb.sqlite (default: the harness's newest backup)")
    ap.add_argument("--limit", type=int, help="only the first N marked tracks, for a quick run")
    ap.add_argument("-j", "--jobs", type=int, help="worker processes (default: one per CPU)")
    args = ap.parse_args(argv)
    cfg = host.load_config()
    path = Path(args.db).expanduser() if args.db else b.newest_backup(cfg)
    if path.resolve() == cfg.mixxx_db.resolve():
        sys.exit("That's Mixxx's live database; pass a copy (the harness's backups are copies)")
    con = db.connect(path)
    try:
        tracks = b.marked_tracks(con)[: args.limit]
    finally:
        con.close()
    print(f"Database: {path}\n{len(tracks)} marked tracks")
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(prepare, tracks, chunksize=4))
    skips = Counter(r["skip"] for r in results if "skip" in r)
    done = [r for r in results if "skip" not in r]
    print(f"Analyzed {len(done)}, skipped {sum(skips.values())}" + "".join(f"\n  {n} {why}" for why, n in skips.most_common()))
    if len(done) < FOLDS:
        print(f"Need at least {FOLDS} analyzed tracks to cross-validate")
        return 1

    fold = np.random.default_rng(0).permutation(len(done)) % FOLDS
    hits = {"ranker": [0, 0], "2 drops rule": [0, 0]}
    for k in range(FOLDS):
        train = [t for t, f in zip(done, fold) if f != k and len(t["y"])]
        model = fit(np.vstack([t["X"] for t in train]), np.concatenate([t["y"] for t in train]))
        for t in (t for t, f in zip(done, fold) if f == k):
            for name, g in (("ranker", pick(t, score(model, t["X"])) if len(t["y"]) else []), ("2 drops rule", t["rule"])):
                hits[name][0] += b.matched(g, t["cues"], t["beat"])
                hits[name][1] += b.matched(g, t["cues"], 4 * t["beat"])

    cues = sum(len(t["cues"]) for t in done)
    drops = sum(t["drops"] for t in done)
    print(f"\nYour {cues} hot cues; {drops} of them count as drops (a cue 8 bars before them)")
    print(f"  drops within a beat of a candidate (the most any ranker can find): {b.pct(sum(t['reachable'] for t in done), drops)}")
    print(f"\n{'Guesses (' + str(FOLDS) + '-fold, by track)':<30}{'cues hit within 1 beat':>24}{'within 1 bar':>14}")
    for name, (beat, bar) in hits.items():
        print(f"{name:<30}{b.pct(beat, cues):>24}{b.pct(bar, cues):>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
