"""The mod's dashboard tab: where guessed cues stand, the tracks waiting for your review, and a
button that runs the guess command. Reads the database only; writing guesses into Mixxx stays
with the harness launch (Mixxx closed)."""
import sys

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QLabel, QListWidget, QPlainTextEdit, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from mixxx_harness import db

import mixxx_cue_guess as cg

REFRESH_S = 5
_HOW = (f"Guessed cues are blue and labelled '{cg.GUESS_LABEL}'. In Mixxx, check each track in "
        f"'{cg.REVIEW_CRATE}', fix any wrong cues, then remove it from the crate: that approves it. "
        "The next launch turns its cues orange, and cratesync then counts it as marked.")


class Page(QWidget):
    def __init__(self, db_path, folder):
        super().__init__()
        self._db_path, self._folder = db_path, folder
        how = QLabel(_HOW)
        how.setWordWrap(True)
        self.counts = QFormLayout()
        self._values = {}
        for key, text in (("waiting", "Guessed, written at the next launch:"), ("in review", "In review:"),
                          ("approved, not recolored yet", "Approved, turn orange at the next launch:"),
                          ("done (left the crate)", "Done (left the crate):"), ("to guess", "Not guessed yet:"),
                          ("need analysis in Mixxx first", "No BPM yet (analyse them in Mixxx first):")):
            self._values[key] = QLabel("-")
            self.counts.addRow(text, self._values[key])
        self.limit = QSpinBox(minimum=0, maximum=5000, value=0)
        self.limit.setSpecialValueText("All")  # shown at 0
        self.limit.setAccessibleName("Guess at most this many tracks")
        self.run_button = QPushButton("Guess cues")
        self.run_button.clicked.connect(self.run_guess)
        row = QHBoxLayout()
        row.addWidget(QLabel("Tracks to guess:"))
        row.addWidget(self.limit)
        row.addWidget(self.run_button)
        row.addStretch()
        self.output = QPlainTextEdit(readOnly=True)
        self.output.setAccessibleName("Guess command output")
        self.output.setPlaceholderText("Guessing reads the audio of every marked track to learn from, "
                                       "then of each track to guess: a few minutes.")
        self.list = QListWidget()
        self.list.setAccessibleName(cg.REVIEW_CRATE)
        self.error = QLabel()
        layout = QVBoxLayout(self)
        layout.addWidget(how)
        layout.addLayout(self.counts)
        layout.addWidget(self.error)
        layout.addLayout(row)
        layout.addWidget(self.output)
        layout.addWidget(QLabel(f"In '{cg.REVIEW_CRATE}':"))
        layout.addWidget(self.list)
        self._proc = None
        self._timer = QTimer(self, interval=REFRESH_S * 1000, timeout=self.refresh)
        self._timer.start()
        self.refresh()

    def refresh(self):
        try:
            con = db.connect(self._db_path)
            con.execute("PRAGMA busy_timeout = 0")
            try:
                st = cg.status(con, self._folder)
                names = {tid: f"{a or '?'} - {t or '?'}" for tid, a, t in con.execute(
                    "SELECT l.id, l.artist, l.title FROM library l JOIN crate_tracks ct ON ct.track_id = l.id "
                    "JOIN crates c ON c.id = ct.crate_id WHERE c.name = ?", (cg.REVIEW_CRATE,))}
            finally:
                con.close()
        except Exception as e:  # locked or missing database: say so, try again at the next refresh
            self.error.setText(f"Can't read the database right now: {e}")
            return
        self.error.setText("")
        for key, label in self._values.items():
            label.setText(str(st[key]))
        wanted = sorted(names.get(tid, path) for tid, path in st["review list"])
        if wanted != [self.list.item(i).text() for i in range(self.list.count())]:
            self.list.clear()
            self.list.addItems(wanted)

    def run_guess(self):
        args = ["-m", "mixxx_cue_guess.guess", "-j", "3"] + (["--limit", str(self.limit.value())] if self.limit.value() else [])
        self.output.clear()
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(
            lambda: self.output.appendPlainText(bytes(self._proc.readAllStandardOutput()).decode(errors="replace").rstrip()))
        self._proc.finished.connect(self._done)
        self.run_button.setEnabled(False)
        self.run_button.setText("Guessing…")
        self._proc.start(sys.executable, ["-u", "-W", "ignore::RuntimeWarning", *args])  # numpy nan warnings: noise here

    def _done(self, code, _status):
        self.run_button.setEnabled(True)
        self.run_button.setText("Guess cues")
        self.output.appendPlainText("Done." if code == 0 else f"Stopped with exit code {code}.")
        self.refresh()
