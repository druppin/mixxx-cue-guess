import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixxx_cue_guess import baseline, ranker
from test_baseline import BEAT, BPM, bar, synth


class Labels(unittest.TestCase):
    def test_drops_are_cues_with_a_cue_8_bars_before(self):
        cues = [10.0, 10 + 32 * BEAT, 100.0, 100 + 32 * BEAT]
        self.assertEqual(ranker.drop_cues(cues, BEAT), [cues[1], cues[3]])
        self.assertEqual(ranker.drop_cues([1.0, 50.0], BEAT), [1.0, 50.0])  # no pattern: all cues


class OnSynthetic(unittest.TestCase):
    def test_candidate_at_the_rise_and_ranker_learns_it(self):
        wav = Path(tempfile.mkdtemp()) / "track.wav"
        synth(wav)
        total, bass, fps = baseline.envelopes(str(wav))
        beats, idx, X = ranker.candidates(total, bass, fps, BPM, *ranker.spectra(str(wav)))
        self.assertEqual(X.shape, (len(idx), len(ranker.FEATURES)))
        times = beats[idx] / fps
        y = (np.abs(times - bar(16)) <= BEAT).astype(float)
        self.assertEqual(y.sum(), 1)
        model = ranker.fit(np.vstack([X] * 4), np.concatenate([y] * 4))
        t = {"beats": beats / fps, "idx": idx}
        g = ranker.pick(t, ranker.score(model, X))
        self.assertTrue(any(abs(x - bar(16)) <= BEAT for x in g))
        self.assertTrue(any(abs(x - bar(8)) <= BEAT for x in g))


if __name__ == "__main__":
    unittest.main()
