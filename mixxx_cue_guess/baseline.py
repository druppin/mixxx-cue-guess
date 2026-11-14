"""Baseline check: how close does a plain rule, with no training, get to the hot cues you set?

    python -m mixxx_cue_guess.baseline              # the harness's newest backup
    python -m mixxx_cue_guess.baseline --db copy.sqlite -v

Reads only: a copy of mixxxdb.sqlite and your audio files. Writes nothing.

For each marked track (4+ hot cues) with a BPM in Mixxx:
1. Beat grid: Mixxx's BPM, with the phase found from the kick drum's onsets.
2. Phrase grid: every 8 bars (32 beats), at the offset where the track changes most.
3. Change at a point: how much the loudness and the bass differ, 8 bars before vs. after.
4. Rules: "top 4" (the 4 biggest changes), "start + top 3" (first beat, then the 3 biggest),
   and "2 drops" (the 2 biggest rises in loudness and bass, each with a cue 8 bars before it:
   the pattern in the first run, 2026-11-13, where 17 of 19 tracks had cues A, A+8 bars, B, B+8 bars).
   Drops are found with 8-bar windows, moved to where a 1-bar window peaks (within 2 bars),
   and never in the first 16 bars. On all 433 tracks: 42% of cues within a beat, up from 14%.
It then counts how many of your cues a guess lands near, and how many of your cues sit
on the beat grid and on the phrase grid at all (if those are low, a better picker can't help).
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from mixxx_harness import db, host

MARKED = 4  # cratesync's rule: a track counts as marked at 4 or more hot cues
HOP = 512  # samples per analysis frame
PHRASE = 32  # beats: 8 bars of 4/4
BASS_HZ = 150
EPS = 1e-10
RULES = ("top 4", "start + top 3", "2 drops")
INTRO = 2 * PHRASE  # beats: no drop in the first 16 bars (the first run's earliest drop was ~20 bars in)
ON_BEAT = 0.1  # of a beat: how close a cue must be to count as on the grid (~45 ms at 130 BPM)


def marked_tracks(con) -> list[dict]:
    """Marked tracks with their path, BPM and hot cue times in seconds."""
    out = []
    for tid, n in db.hot_cue_counts(con).items():
        if n < MARKED:
            continue
        rate, channels = db.audio_format(con, tid)
        bpm = con.execute("SELECT bpm FROM library WHERE id = ?", (tid,)).fetchone()[0]
        # Cue positions count samples across all channels (cratesync findings, spike 1).
        cues = sorted(pos / (rate * channels) for _, pos in db.hot_cues(con, tid)) if rate and channels else []
        out.append({"id": tid, "path": db.track_path(con, tid), "bpm": bpm, "cues": cues})
    return out


def envelopes(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Per frame: log energy, log bass energy; and frames per second."""
    info = sf.info(path)
    total, bass = [], []
    for block in sf.blocks(path, blocksize=HOP * 4096, dtype="float32", always_2d=True):
        x = block.mean(axis=1)
        frames = x[: len(x) // HOP * HOP].reshape(-1, HOP)
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        cut = int(BASS_HZ * HOP / info.samplerate) + 1
        total.append(power.sum(axis=1))
        bass.append(power[:, 1:cut].sum(axis=1))
    return np.log(np.concatenate(total) + EPS), np.log(np.concatenate(bass) + EPS), info.samplerate / HOP


def beat_frames(bass: np.ndarray, period: float) -> np.ndarray:
    """Frames of every beat: the phase whose beats land on the most bass onsets."""
    onset = np.maximum(np.diff(bass, prepend=bass[0]), 0)
    phases = np.arange(int(np.ceil(period)))
    count = int((len(bass) - 1) / period)
    scores = [onset[np.round(p + period * np.arange(count)).astype(int).clip(0, len(bass) - 1)].sum() for p in phases]
    first = phases[int(np.argmax(scores))]
    return np.round(first + period * np.arange(int((len(bass) - 1 - first) / period) + 1)).astype(int)


def window_mean(v: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(v)])
    s, e = starts.clip(0, len(v)), ends.clip(0, len(v))
    return np.where(e > s, (c[e] - c[s]) / np.maximum(e - s, 1), np.nan)


def change(total: np.ndarray, bass: np.ndarray, beats: np.ndarray, width: int) -> np.ndarray:
    """At each beat: |after - before| of mean log energy and bass over `width` frames each side."""
    out = np.zeros(len(beats))
    for v in (total, bass):
        before, after = window_mean(v, beats - width, beats), window_mean(v, beats, beats + width)
        out += np.nan_to_num(np.abs(after - before))  # no audio on one side (track edges): no change
    return out


def rise(total: np.ndarray, bass: np.ndarray, beats: np.ndarray, width: int) -> np.ndarray:
    """At each beat: after - before, like change() but signed, so a drop scores high and an outro low.
    Beats without a full window on both sides score -inf: else the silence before the first beat reads as a drop."""
    out = np.zeros(len(beats))
    for v in (total, bass):
        out += window_mean(v, beats, beats + width) - window_mean(v, beats - width, beats)
    out[(beats < width) | (beats + width > len(total))] = -np.inf
    return np.nan_to_num(out, nan=-np.inf)


def sharpen(r: np.ndarray, d: int, reach: int = 8) -> int:
    """The beat within `reach` of d where the short-window rise r peaks: the 8-bar window finds the drop, but blurs it by a bar or two."""
    lo = max(d - reach, 0)
    return lo + int(np.argmax(r[lo:d + reach + 1]))


def drops(r: np.ndarray, n: int = 2, apart: int = 2 * PHRASE) -> list[int]:
    """The n beats with the biggest rise, at least `apart` beats from each other."""
    picked = []
    for i in np.argsort(-r, kind="stable"):
        if all(abs(int(i) - p) >= apart for p in picked):
            picked.append(int(i))
            if len(picked) == n:
                break
    return picked


def guess(total, bass, fps: float, bpm: float) -> dict:
    period = fps * 60 / bpm
    beats = beat_frames(bass, period)
    score = change(total, bass, beats, int(round(period * PHRASE)))
    offset = max(range(min(PHRASE, len(beats))), key=lambda o: score[o::PHRASE].sum())
    phrase = np.arange(offset, len(beats), PHRASE)
    top = phrase[np.argsort(-score[phrase], kind="stable")]
    seconds = lambda idx: sorted(float(beats[i] / fps) for i in idx)  # noqa: E731
    short = rise(total, bass, beats, int(round(period * 4)))
    long = rise(total, bass, beats, int(round(period * PHRASE)))
    long[:INTRO] = -np.inf  # the kick coming in after the intro rises too, but you cue drops, not intros
    found = [sharpen(short, d) for d in drops(long)]
    return {
        "beats": beats / fps,
        "phrase": beats[phrase] / fps,
        "top 4": seconds(top[:4]),
        "start + top 3": seconds([0] + [i for i in top if i != 0][:3]),
        "2 drops": seconds(found + [max(d - PHRASE, 0) for d in found]),
    }


def matched(guesses, cues, tol: float) -> int:
    """How many cues a guess lands within tol of, each guess and cue used once (closest first)."""
    pairs = sorted((abs(g - c), i, j) for i, g in enumerate(guesses) for j, c in enumerate(cues) if abs(g - c) <= tol)
    used_g, used_c = set(), set()
    for _, i, j in pairs:
        if i not in used_g and j not in used_c:
            used_g.add(i), used_c.add(j)
    return len(used_c)


def near(points: np.ndarray, cues, tol: float) -> int:
    return sum(bool(len(points)) and float(np.min(np.abs(points - c))) <= tol for c in cues)


def analyze(track: dict) -> dict:
    """One track's result, or {"skip": reason}. Runs in a worker process."""
    if not track["path"]:
        return {"skip": "no file path"}
    if not track["bpm"] or track["bpm"] <= 0:
        return {"skip": "no BPM in Mixxx"}
    if not track["cues"]:
        return {"skip": "no sample rate in Mixxx"}
    if not Path(track["path"]).is_file():
        return {"skip": "file not found"}
    try:
        total, bass, fps = envelopes(track["path"])
    except Exception as e:  # e.g. a format libsndfile can't read
        return {"skip": f"can't decode ({Path(track['path']).suffix or 'no extension'})", "error": str(e)}
    g = guess(total, bass, fps, track["bpm"])
    beat, cues = 60 / track["bpm"], track["cues"]
    out = {"cues": len(cues), "on beat grid": near(g["beats"], cues, beat * ON_BEAT),
           "on phrase grid": near(g["phrase"], cues, beat)}
    for rule in RULES:
        out[rule] = {"beat": matched(g[rule], cues, beat), "bar": matched(g[rule], cues, 4 * beat), "guesses": g[rule]}
    return out


def newest_backup(cfg) -> Path:
    found = sorted(cfg.backup_dir.glob("mixxxdb-*.sqlite"))
    if not found:
        sys.exit(f"No backups in {cfg.backup_dir}; launch Mixxx through the harness once, or pass --db")
    return found[-1]


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", help="a copy of mixxxdb.sqlite (default: the harness's newest backup)")
    ap.add_argument("--limit", type=int, help="only the first N marked tracks, for a quick run")
    ap.add_argument("-j", "--jobs", type=int, help="worker processes (default: one per CPU)")
    ap.add_argument("-v", "--verbose", action="store_true", help="one line per track")
    args = ap.parse_args(argv)
    cfg = host.load_config()
    path = Path(args.db).expanduser() if args.db else newest_backup(cfg)
    if path.resolve() == cfg.mixxx_db.resolve():
        sys.exit("That's Mixxx's live database; pass a copy (the harness's backups are copies)")
    con = db.connect(path)
    try:
        tracks = marked_tracks(con)[: args.limit]
    finally:
        con.close()
    print(f"Database: {path}\n{len(tracks)} marked tracks")
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(analyze, tracks, chunksize=4))

    skips = Counter(r["skip"] for r in results if "skip" in r)
    done = [(t, r) for t, r in zip(tracks, results) if "skip" not in r]
    if args.verbose:
        for t, r in zip(tracks, results):
            name = Path(t["path"] or "?").name
            if "skip" in r:
                print(f"  {t['id']:>5} {name}: skipped, {r['skip']} {r.get('error', '')}")
            else:
                cues = " ".join(f"{c:.1f}" for c in t["cues"])
                best = " ".join(f"{c:.1f}" for c in r[RULES[-1]]["guesses"])
                print(f"  {t['id']:>5} {name}: yours {cues} | {RULES[-1]} {best} | {r[RULES[-1]]['beat']}/{r['cues']} within a beat")
    print(f"Analyzed {len(done)}, skipped {sum(skips.values())}" + "".join(f"\n  {n} {why}" for why, n in skips.most_common()))
    if not done:
        return 1
    cues = sum(r["cues"] for _, r in done)
    print(f"\nYour {cues} hot cues on these tracks:")
    print(f"  within {ON_BEAT:g} beat of the beat grid: {pct(sum(r['on beat grid'] for _, r in done), cues)}")
    print(f"  within a beat of the 8-bar phrase grid: {pct(sum(r['on phrase grid'] for _, r in done), cues)}")
    print(f"\n{'Rule':<16}{'cues hit within 1 beat':>24}{'within 1 bar':>14}")
    for rule in RULES:
        beat = sum(r[rule]["beat"] for _, r in done)
        bar = sum(r[rule]["bar"] for _, r in done)
        print(f"{rule:<16}{pct(beat, cues):>24}{pct(bar, cues):>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
