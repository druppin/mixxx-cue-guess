# mixxx-cue-guess

A [mixxx-harness](https://github.com/druppin/mixxx-harness) mod that guesses where you'd put
hot cues, based on the hot cues you've already set.

Status: **baseline check and a trained ranker, both report-only.** The mod writes nothing yet. Writing cues into Mixxx needs a
spike first (which `type` and `color` a new cue needs, and whether Mixxx keeps cues written
from outside).

## Baseline check

```
uv run python -m mixxx_cue_guess.baseline           # the harness's newest database backup
uv run python -m mixxx_cue_guess.baseline --db ~/copy-of-mixxxdb.sqlite -v --limit 20
```

It reads a copy of `mixxxdb.sqlite` (it refuses the live one) and your audio files, and
writes nothing. For each marked track (4+ hot cues) with a BPM in Mixxx:

1. **Beat grid:** Mixxx's BPM, with the phase taken from kick-drum onsets. (Mixxx's own
   beatgrid blob isn't read: its format isn't verified yet.)
2. **Phrase grid:** every 8 bars, at the offset where the track changes the most.
3. **Change:** how different loudness and bass are in the 8 bars before and after a point.
4. **Rules, no training:** "top 4" biggest changes, and "start + top 3".

The report shows what share of your cues sit on the beat grid and on the phrase grid (the most
any phrase-based picker could reach), and what share each rule hits within a beat and within
a bar. A trained model has to beat these numbers to be worth building.

Audio goes through libsndfile (WAV, FLAC, MP3, OGG). Other formats, such as M4A, are skipped
and counted.

## Trained ranker

```
uv run python -m mixxx_cue_guess.ranker             # same --db / --limit / -j options
```

Learns which rises in a track you cue as drops. Candidates are the beats where loudness and
bass rise most within 2 bars either side, plus every 8-bar phrase start (the phrase phase is the
one where the 1-bar change adds up highest). Each gets a few features (how big the rise is over 1
and 8 bars, compared with the track's biggest, where it sits in the track, loudness before and
after, fit to the phrase grid, whether there's another big rise 16-32 bars away). A
gradient-boosting model (scikit-learn) scores them. The guess is the 2 best candidates 16+ bars
apart, each with a cue 8 bars before it.

It's scored with 5-fold cross-validation by track, repeated over 5 different splits, so every
track is guessed by a model that never saw it, next to the baseline's "2 drops" rule. On the
library (about 430 tracks, 2026-11-13): **54%** of hot cues within a beat (62% within a bar),
against 42% (52%) for the rule. About 82% of drop cues sit within a beat of a candidate.

Tried and not kept (same tracks, averaged over splits):
- Mixxx's own beatgrid (decoded from the `beats` blob): 94% of cues sit on it against 83% on
  this script's grid, but the score didn't move.
- Every beat as a candidate; picking the best-scoring pair of drops; stricter labels; wider
  drop spacing; bigger or smaller models: all the same or worse.
- Mid and high frequency bands as features: +0.4 points, within the noise between splits.
- Each feature compared with the rest of its track, features of the build 8 bars earlier, and
  weighting every track equally in training: all within ±0.7 points of the shipped version.

It writes nothing and saves no model yet; that comes with writing cues into Mixxx.

## Tests

`uv run python -m unittest discover tests` uses a synthetic track and a throwaway database
(`mixxx_harness.testing.make_db`), never a real library.
