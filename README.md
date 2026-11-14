# mixxx-cue-guess

A [mixxx-harness](https://github.com/druppin/mixxx-harness) mod that guesses where you'd put
hot cues, based on the hot cues you've already set.

Status: **baseline check only.** The mod writes nothing yet. Writing cues into Mixxx needs a
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

## Tests

`uv run python -m unittest discover tests` uses a synthetic track and a throwaway database
(`mixxx_harness.testing.make_db`), never a real library.
