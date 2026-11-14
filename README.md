# mixxx-cue-guess

A [mixxx-harness](https://github.com/druppin/mixxx-harness) mod that guesses where you'd put
hot cues, based on the hot cues you've already set.

Status: **writes guessed hot cues for you to review** (2026-11-14).

## Guessed cues, and approving them

1. `uv run python -m mixxx_cue_guess.guess` trains the ranker (below) on your marked tracks and
   guesses 4 hot cues for every track with none: 2 drops, each with a cue 8 bars before it,
   on pads 1-4 in time order. It reads a copy of the database and your audio, and writes only
   `<harness state_dir>/cue-guess/guesses.json`. Options: `--limit N`, `-j N` (default 3), `--db`.
2. At the next harness launch (Mixxx closed), the mod writes them into Mixxx: hot cues colored
   **blue** (`0x0044FF`) and labelled `guess`, and the track goes into the crate **Cues: review**.
   A track that has any hot cues is never written to, nor is one whose file changed since the
   guess.
   On tracks in Cues: review it also sets Mixxx's **intro end** and **outro start**, where you
   haven't set them (see "Intro and outro" below).
3. Check the cues (and intro/outro) in Mixxx and fix any that are wrong. Then **remove the track from Cues:
   review**: that's the approval. At the next launch the mod turns its `guess` cues orange
   (`0xFF8000`, your usual color) and clears the label, so blue on a deck always means "not
   checked yet".
The harness dashboard (`python -m mixxx_harness dashboard`) has a **cue-guess** tab: how many
tracks are guessed, in review, approved and still to analyse; the tracks in Cues: review; and a
**Guess cues** button that runs step 1 (with an optional batch size) and shows its output.

4. Don't want them? Delete the cues. The mod lists every track it wrote to
   (`cue-guess/written.json`) and never guesses one twice.

cratesync should treat tracks in Cues: review as not marked (add the crate to the bridge's
ignore list). The mod claims `Owns(crates={"Cues: review"}, new_cues=True)` in mixxx-harness:
it can add hot cues only to tracks with none, and change cue colors and labels, never positions.
For the harness to run it, install this package in the environment that runs
`python -m mixxx_harness launch`.

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

## Intro and outro

`uv run python -m mixxx_cue_guess.markers` scores it. On marked tracks you set the intro end and
outro start almost always (418 and 411 of 438, 2026-11-14), on 8-bar steps from your cues: the
intro end on your first hot cue or 8-24 bars before it, the outro start 8-48 bars after your
last. So the candidates are those steps from the guessed first and last cue, and a
gradient-boosting model (the drop ranker's per-beat features, plus how many phrases away)
picks one of each. Scored 3x5-fold by track (about 380 tracks, 2026-11-14), within a beat:
intro end 50%, outro start 47% from guessed cues; 66% and 63% from your real cues, against 34%
and 32% for a fixed rule (8 bars before the first cue, 16 after the last). The mod writes them only where the marker isn't set, only on tracks in
Cues: review, and never changes one you set (mixxx-harness refuses it).

Mixxx's rows, worked out from the library (not in cratesync's findings): intro type 6, start
`position`, end `position + length` (length 0 = no end); outro type 7, `position` -1 with the
end in `length` when no start is set, else start `position`, end `position + length`.
**Not checked yet: that Mixxx shows markers written this way.** Look at one track in Mixxx
after the first launch that writes them.

## Trained ranker

```
uv run python -m mixxx_cue_guess.ranker             # same --db / --limit / -j options
```

Learns which rises in a track you cue as drops. Candidates are the beats where loudness and
bass rise most within 2 bars either side, plus every 8-bar phrase start (the phrase phase is the
one where the 1-bar change adds up highest). Each gets a few features (how big the rise is over 1
and 8 bars, compared with the track's biggest, where it sits in the track, loudness before and
after, fit to the phrase grid, whether there's another big rise 16-32 bars away; and from a
mel spectrogram and chroma, novelty, timbre change and harmony change across it). A
gradient-boosting model (scikit-learn) scores them. The guess is the 2 best candidates 16+ bars
apart, each with a cue 8 bars before it.

It's scored with 5-fold cross-validation by track, repeated over 5 different splits, so every
track is guessed by a model that never saw it, next to the baseline's "2 drops" rule. On the
library (about 400-430 tracks, 2026-11-14): **55-56%** of hot cues within a beat (64% within a
bar), against 42% (52%) for the rule. About 81% of drop cues sit within a beat of a candidate.
Analysis reads each file twice (once for loudness and bass, once for the mel spectrogram).

Tried and not kept (same tracks, averaged over splits):
- Mixxx's own beatgrid (decoded from the `beats` blob): 94% of cues sit on it against 83% on
  this script's grid, but the score didn't move.
- Every beat as a candidate; picking the best-scoring pair of drops; stricter labels; wider
  drop spacing; bigger or smaller models: all the same or worse.
- Mid and high frequency bands as features: +0.4 points, within the noise between splits.
- Spectrogram models (2026-11-14): an MLP on a 16-bar mel patch around each candidate (46% alone),
  a small CNN on the same patch (51% alone; 54.7% blended 50/50 with gradient boosting), and the
  MLP's score as an extra feature (no gain). How alike the sections after two candidates sound
  (repetition) didn't help either.
- Each feature compared with the rest of its track, features of the build 8 bars earlier, and
  weighting every track equally in training: all within ±0.7 points of the shipped version.

It writes nothing and saves no model yet; that comes with writing cues into Mixxx.

## Tests

`uv run python -m unittest discover tests` uses a synthetic track and a throwaway database
(`mixxx_harness.testing.make_db`), never a real library.
