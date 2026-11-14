"""Intro end and outro start: learned from the markers you set, placed relative to the cues.

    python -m mixxx_cue_guess.markers             # cross-validated score on your library

On marked tracks you set Mixxx's intro end and outro start almost always (418 and 411 of 438,
2026-11-14), and on 8-bar steps from your cues: the intro end on your first hot cue or 8, 16 or
24 bars before it (87%), the outro start 8 to 48 bars after your last (82%). So the candidates
are those steps from the (guessed) first and last cue, scored by a gradient-boosting model on
the same per-beat features as the drop ranker.

Mixxx's rows (worked out from the library, 2026-11-14; not in cratesync's findings): intro is
type 6, starting at `position`, ending at `position + length` (length 0: no end set); outro is
type 7, `position` -1 with the end in `length` when no start is set, else start `position`,
end `position + length`. Positions count samples across all channels.
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mixxx_cue_guess import baseline as b, ranker
from mixxx_harness import db, host

INTRO_STEPS = range(0, 7)  # phrases (8 bars) before the first cue
OUTRO_STEPS = range(1, 9)  # phrases after the last cue


def marked_tracks(con) -> list[dict]:
    """baseline.marked_tracks, plus the intro end and outro start you set (seconds, or None)."""
    out = b.marked_tracks(con)
    for t in out:
        rate, channels = db.audio_format(con, t["id"])
        s = (rate or 0) * (channels or 0)
        intro = con.execute("SELECT position + length FROM cues WHERE track_id = ? AND type = 6 AND length > 0",
                            (t["id"],)).fetchone()
        outro = con.execute("SELECT position FROM cues WHERE track_id = ? AND type = 7 AND position >= 0",
                            (t["id"],)).fetchone()
        t["intro"] = intro[0] / s if intro and s else None
        t["outro"] = outro[0] / s if outro and s else None
    return out


def lattice(anchor: int, n: int, steps, sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat indices anchor + sign * 32 * k inside the track, and their k."""
    k = np.array([k for k in steps if 0 <= anchor + sign * b.PHRASE * k < n], dtype=int)
    return anchor + sign * b.PHRASE * k, k


def rows(Xall: np.ndarray, idx: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.column_stack([Xall[idx], k, idx / len(Xall)]) if len(idx) else np.zeros((0, Xall.shape[1] + 2))


def nearest_beat(beats_s: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(beats_s - t)))


def candidates(Xall, beats_s, cues: list[float]):
    """For the intro end and the outro start: (beat indices, feature rows), from the first and last cue."""
    first, last = nearest_beat(beats_s, cues[0]), nearest_beat(beats_s, cues[-1])
    out = {}
    for name, anchor, steps, sign in (("intro", first, INTRO_STEPS, -1), ("outro", last, OUTRO_STEPS, 1)):
        idx, k = lattice(anchor, len(beats_s), steps, sign)
        out[name] = (idx, rows(Xall, idx, k))
    return out


def all_features(A: dict) -> np.ndarray:
    return ranker.features(A, np.arange(len(A["beats"])))


def pick(model, Xall, beats_s, cues: list[float]) -> dict:
    """{"intro": seconds or None, "outro": seconds or None} from guessed (or real) cues."""
    out = {}
    for name, (idx, X) in candidates(Xall, beats_s, cues).items():
        m = model[name]
        out[name] = float(beats_s[idx[int(np.argmax(m.predict_proba(X)[:, 1]))]]) if m and len(idx) else None
    return out


def fit(tracks: list[dict]) -> dict:
    """An intro and an outro model, trained from your real cues and markers (None: no markers to learn from)."""
    out = {}
    for name in ("intro", "outro"):
        Xs, ys = [], []
        for t in tracks:
            if t[name] is None:
                continue
            idx, X = candidates(t["Xall"], t["beats"], t["cues"])[name]
            Xs.append(X)
            ys.append(np.abs(t["beats"][idx] - t[name]) <= t["beat"])
        y = np.concatenate(ys).astype(float) if ys else np.zeros(0)
        out[name] = ranker.fit(np.vstack(Xs), y) if 0 < y.sum() < len(y) else None  # nothing to learn: no guess
    return out


def prepare(track: dict) -> dict:
    """ranker.prepare's drop data plus every beat's features, or {"skip": reason}. Runs in a worker."""
    loaded = b.load(track)
    if isinstance(loaded, dict):
        return loaded
    total, bass, fps = loaded
    try:
        mel, chroma = ranker.spectra(track["path"])
    except Exception as e:
        return {"skip": f"can't decode ({Path(track['path']).suffix or 'no extension'})", "error": str(e)}
    A = ranker.analysis(total, bass, fps, track["bpm"], mel, chroma)
    beats, idx, X = ranker.drop_candidates(A)
    beat = 60 / track["bpm"]
    times = beats[idx] / fps
    labels = ranker.drop_cues(track["cues"], beat)
    return {"X": X, "y": np.array([any(abs(t - c) <= beat for c in labels) for t in times], dtype=float),
            "idx": idx, "beats": beats / fps, "cues": track["cues"], "beat": beat, "Xall": all_features(A),
            "intro": track.get("intro"), "outro": track.get("outro")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="a copy of mixxxdb.sqlite (default: the harness's newest backup)")
    ap.add_argument("--limit", type=int, help="only the first N marked tracks, for a quick run")
    ap.add_argument("-j", "--jobs", type=int, default=3, help="worker processes (default 3)")
    ap.add_argument("--splits", type=int, default=3, help="cross-validation repeats (default 3)")
    args = ap.parse_args(argv)
    cfg = host.load_config()
    path = Path(args.db).expanduser() if args.db else b.newest_backup(cfg)
    if path.resolve() == cfg.mixxx_db.resolve():
        sys.exit("That's Mixxx's live database; pass a copy (the harness's backups are copies)")
    con = db.connect(path)
    try:
        tracks = marked_tracks(con)[: args.limit]
    finally:
        con.close()
    print(f"Database: {path}\n{len(tracks)} marked tracks")
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(prepare, tracks, chunksize=4))
    skips = Counter(r["skip"] for r in results if "skip" in r)
    done = [r for r in results if "skip" not in r and len(r["y"])]
    print(f"Analyzed {len(done)}, skipped {len(results) - len(done)}" + "".join(f"\n  {n} {why}" for why, n in skips.most_common()))
    if len(done) < 5:
        return 1

    # name -> way -> [within a beat, within a bar, tracks with that marker]
    score = {n: {w: [0, 0, 0] for w in ("model, guessed cues", "model, your cues", "rule, your cues")} for n in ("intro", "outro")}
    rule = {"intro": lambda t: t["cues"][0] - 32 * t["beat"], "outro": lambda t: t["cues"][-1] + 64 * t["beat"]}
    for seed in range(args.splits):
        fold = np.random.default_rng(seed).permutation(len(done)) % 5
        for k in range(5):
            train = [t for t, f in zip(done, fold) if f != k]
            cue_model = ranker.fit(np.vstack([t["X"] for t in train]), np.concatenate([t["y"] for t in train]))
            models = fit(train)
            for t in (t for t, f in zip(done, fold) if f == k):
                guessed = ranker.pick(t, ranker.score(cue_model, t["X"]))
                ways = {"model, your cues": pick(models, t["Xall"], t["beats"], t["cues"]),
                        "rule, your cues": {n: rule[n](t) for n in rule}}
                ways["model, guessed cues"] = pick(models, t["Xall"], t["beats"], guessed) if len(guessed) == 4 else {}
                for n in ("intro", "outro"):
                    if t[n] is None:
                        continue
                    for w, g in ways.items():
                        s = score[n][w]
                        s[2] += 1
                        if g.get(n) is not None:
                            s[0] += abs(g[n] - t[n]) <= t["beat"]
                            s[1] += abs(g[n] - t[n]) <= 4 * t["beat"]
    print(f"\n{'Marker':<8}{f'Guess ({args.splits}x5-fold, by track)':<34}{'within 1 beat':>15}{'within 1 bar':>14}")
    for n, ways in score.items():
        for w, (beat, bar, total) in ways.items():
            print(f"{n:<8}{w:<34}{b.pct(beat, total):>15}{b.pct(bar, total):>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
