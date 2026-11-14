"""Trained ranker: learns which rises in a track you cue as drops, scored on tracks it didn't train on.

    python -m mixxx_cue_guess.ranker              # the harness's newest backup
    python -m mixxx_cue_guess.ranker --db copy.sqlite

Reads only: a copy of mixxxdb.sqlite and your audio files. Writes nothing.

1. Candidates: every beat where the 1-bar rise peaks within 2 bars either side (the point
   the baseline's "2 drops" rule sharpens a drop to), plus every 8-bar phrase start.
2. Labels: a candidate is a drop if it's within a beat of one of your cues that has another of
   your cues 8 bars before it (your A, A+8 bars pattern); on tracks without that pattern, any cue.
3. Model: gradient boosting on a few features per candidate (rise sizes, place in the track,
   loudness after vs. before, fit to the track's phrase grid, a big rise 16-32 bars away), and
   from a mel spectrogram and chroma: novelty, timbre change and harmony change across it.
4. Guess: the 2 top-scored candidates 16+ bars apart, each with a cue 8 bars before it.
Scored with 5-fold cross-validation by track, averaged over 5 different splits (one split
alone moves the score by up to 2 points), against the "2 drops" rule on the same tracks.
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier

from mixxx_cue_guess import baseline as b
from mixxx_harness import db, host

FOLDS = 5
SPLITS = 5  # cross-validation repeats, each with a different split
FEATURES = ("8-bar rise", "1-bar rise", "rise vs. track's biggest", "place in track", "bars from start",
            "loudness after", "bass after", "loudness before", "on the phrase grid",
            "rise rank in track", "1-bar peak", "big rise 16-32 bars away", "bass before",
            "beats from the grid's phrase start", "1-bar change",
            "novelty over 8 bars", "novelty over 2 bars",
            *(f"timbre {w}: {n}" for w in ("8 bars", "1 bar") for n in ("change", "low", "low-mid", "mid", "high-mid", "high")),
            "brightness after", "harmony change over 8 bars", "harmony change over 1 bar")
SPEC_N, SPEC_HOP, MELS = 2048, 1024, 40  # finer FFT than baseline.envelopes: mel and chroma need the resolution


def banks(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Mel (40 triangles, 30 Hz-16 kHz) and chroma (12 pitch classes, 60 Hz-4 kHz) filters over FFT bins."""
    hz = np.fft.rfftfreq(SPEC_N, 1 / sr)
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    edges = 700 * (10 ** (np.linspace(mel(30), mel(min(16000, sr / 2)), MELS + 2) / 2595) - 1)
    fbm = np.array([np.clip(np.minimum((hz - lo) / (c - lo), (hi - hz) / (hi - c)), 0, None)
                    for lo, c, hi in zip(edges, edges[1:], edges[2:])])
    fbc = np.zeros((12, len(hz)))
    ok = (hz > 60) & (hz < 4000)
    fbc[np.round(12 * np.log2(hz[ok] / 440) + 69).astype(int) % 12, np.flatnonzero(ok)] = 1
    return fbm, fbc


def spectra(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Per SPEC_HOP frame: log mel energies, and chroma as shares of the frame's pitched energy.

    shortcut: decodes the file a second time after baseline.envelopes; merge the two passes if analysis time matters.
    """
    sr = sf.info(path).samplerate
    fbm, fbc = banks(sr)
    win = np.hanning(SPEC_N).astype(np.float32)
    mel, chroma = [], []
    for block in sf.blocks(path, blocksize=SPEC_HOP * 2048, overlap=SPEC_N - SPEC_HOP, dtype="float32", always_2d=True):
        x = block.mean(axis=1)
        if len(x) < SPEC_N:
            break
        p = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, SPEC_N)[::SPEC_HOP] * win, axis=1)) ** 2
        mel.append(p @ fbm.T)
        chroma.append(p @ fbc.T)
    mel, chroma = np.concatenate(mel), np.concatenate(chroma)
    return np.log(mel + b.EPS), chroma / (chroma.sum(axis=1, keepdims=True) + b.EPS)


def per_beat(frames: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Mean of SPEC_HOP frames over each beat; `beats` is in baseline.HOP frames."""
    edges = np.r_[beats, beats[-1] + np.median(np.diff(beats))] * b.HOP / SPEC_HOP
    edges = np.clip(np.round(edges).astype(int), 0, len(frames))
    c = np.concatenate([np.zeros((1, frames.shape[1])), np.cumsum(frames, axis=0)])
    return (c[edges[1:]] - c[edges[:-1]]) / np.maximum(edges[1:] - edges[:-1], 1)[:, None]


def spectral(mel: np.ndarray, chroma: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per candidate: novelty, timbre change and harmony change across it (each +0.5-1.2 points, 2026-11-14)."""
    Z = (mel - mel.mean(axis=0)) / (mel.std(axis=0) + 1e-6)  # each band against its own track
    U = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    span = lambda A, a, c: A[max(a, 0):min(c, len(A))].mean(axis=0) if min(c, len(A)) > max(a, 0) else np.zeros(A.shape[1])  # noqa: E731
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))  # noqa: E731

    def novelty(i, w):  # Foote: similar within each side, unlike across
        A, B = U[max(i - w, 0):i], U[i:i + w]
        return float((A @ A.T).mean() + (B @ B.T).mean() - 2 * (A @ B.T).mean()) if len(A) and len(B) else 0.0

    rows = []
    for i in idx:
        row = [novelty(i, b.PHRASE), novelty(i, 8)]
        for w in (b.PHRASE, 4):
            d = span(Z, i, i + w) - span(Z, i - w, i)
            row += [np.linalg.norm(d)] + [d[k:k + 8].mean() for k in range(0, MELS, 8)]
        row.append(span(Z, i, i + b.PHRASE)[24:].mean())
        row += [1 - cos(span(chroma, i - w, i), span(chroma, i, i + w)) for w in (b.PHRASE, 4)]
        rows.append(row)
    return np.array(rows).reshape(len(idx), -1)


def candidates(total, bass, fps: float, bpm: float, mel: np.ndarray, chroma: np.ndarray):
    """Beat frames, candidate beat indices, and one feature row per candidate.

    Candidates: the peaks of the 1-bar rise, plus every 8-bar phrase start. The phrase phase is the
    one where the 1-bar change adds up highest (right within a beat on 79% of tracks, 2026-11-13).
    """
    period = fps * 60 / bpm
    beats = b.beat_frames(bass, period)
    width = int(round(period * b.PHRASE))
    long = b.rise(total, bass, beats, width)
    short = b.rise(total, bass, beats, int(round(period * 4)))
    ch = np.nan_to_num(b.change(total, bass, beats, int(round(period * 4))))
    ok = np.flatnonzero(np.isfinite(long) & np.isfinite(short))
    peaks = [i for i in ok if b.sharpen(short, i) == i]
    phase = max(range(b.PHRASE), key=lambda p: ch[p::b.PHRASE].sum())
    idx = np.array(sorted(set(peaks) | {i for i in ok if i % b.PHRASE == phase}), dtype=int)
    if not len(idx):
        return beats, idx, np.zeros((0, len(FEATURES)))
    after = lambda v: b.window_mean(v, beats[idx], beats[idx] + width) - np.median(v)  # noqa: E731
    before = lambda v: b.window_mean(v, beats[idx] - width, beats[idx]) - np.median(v)  # noqa: E731
    off = (idx - phase) % b.PHRASE
    away = [idx + k for k in (-128, -96, -64, 64, 96, 128)]  # 16, 24, 32 bars: where the other drop tends to be
    partner = np.max([np.where((j >= 0) & (j < len(long)), np.nan_to_num(long[j.clip(0, len(long) - 1)], neginf=-5), 0)
                      for j in away], axis=0)
    spec = spectral(per_beat(mel, beats), per_beat(chroma, beats), idx)
    X = np.column_stack([long[idx], short[idx], long[idx] / max(long[idx].max(), b.EPS), idx / len(beats),
                         idx / 4 / 100, after(total), after(bass), before(total), 1 - np.minimum(off, b.PHRASE - off) / 16,
                         np.argsort(np.argsort(-long[idx])) / len(idx), np.isin(idx, peaks), partner, before(bass),
                         np.minimum(idx % b.PHRASE, b.PHRASE - idx % b.PHRASE), ch[idx], spec])
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
    try:
        mel, chroma = spectra(track["path"])
    except Exception as e:
        return {"skip": f"can't decode ({Path(track['path']).suffix or 'no extension'})", "error": str(e)}
    beats, idx, X = candidates(total, bass, fps, track["bpm"], mel, chroma)
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

    hits = {"ranker": [0, 0], "2 drops rule": [0, 0]}
    for seed in range(SPLITS):
        fold = np.random.default_rng(seed).permutation(len(done)) % FOLDS
        for k in range(FOLDS):
            train = [t for t, f in zip(done, fold) if f != k and len(t["y"])]
            model = fit(np.vstack([t["X"] for t in train]), np.concatenate([t["y"] for t in train]))
            for t in (t for t, f in zip(done, fold) if f == k):
                for name, g in (("ranker", pick(t, score(model, t["X"])) if len(t["y"]) else []), ("2 drops rule", t["rule"])):
                    hits[name][0] += b.matched(g, t["cues"], t["beat"]) / SPLITS
                    hits[name][1] += b.matched(g, t["cues"], 4 * t["beat"]) / SPLITS

    cues = sum(len(t["cues"]) for t in done)
    drops = sum(t["drops"] for t in done)
    print(f"\nYour {cues} hot cues; {drops} of them count as drops (a cue 8 bars before them)")
    print(f"  drops within a beat of a candidate (the most any ranker can find): {b.pct(sum(t['reachable'] for t in done), drops)}")
    print(f"\n{f'Guesses ({FOLDS}-fold x{SPLITS}, by track)':<30}{'cues hit within 1 beat':>24}{'within 1 bar':>14}")
    for name, (beat, bar) in hits.items():
        print(f"{name:<30}{b.pct(beat, cues):>24}{b.pct(bar, cues):>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
