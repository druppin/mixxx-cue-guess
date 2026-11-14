"""The mixxx-harness mod. It writes nothing yet: so far there's only the baseline check
(`python -m mixxx_cue_guess.baseline`), which reads a copy of the database."""
from mixxx_harness.mods import Owns


class CueGuess:
    name, version = "cue-guess", "0.1.0"
    owns = Owns()  # nothing until writing cues is spiked

    def plan(self, con):
        return []


mod = CueGuess()
