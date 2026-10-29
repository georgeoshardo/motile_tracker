from __future__ import annotations

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QWidget,
)


class KymographTimeSlider(QWidget):
    """A slider shown under the canvas in kymograph mode for panning along time.

    Each step is one frame: moving the slider centres the view on that frame
    (turning the page when the frame is not on the current one), and panning the
    view moves the slider. The arrow buttons step one frame; so do the Left/Right
    keys while the slider has focus.
    """

    timepoint_changed = Signal(int)

    def __init__(self):
        super().__init__()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        self.slider.setTracking(True)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider.setToolTip(
            "Pan the kymograph along time, one frame per step (Left/Right keys "
            "when focused). The page is turned when the frame is not on it."
        )
        self.slider.valueChanged.connect(self._on_value_changed)

        self.previous_button = QPushButton("◀")
        self.previous_button.setFixedWidth(28)
        self.previous_button.setToolTip("One frame back")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.next_button = QPushButton("▶")
        self.next_button.setFixedWidth(28)
        self.next_button.setToolTip("One frame forward")
        self.next_button.clicked.connect(lambda: self.step(1))

        self.label = QLabel()
        self.label.setMinimumWidth(90)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(QLabel("time"))
        layout.addWidget(self.previous_button)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.next_button)
        layout.addWidget(self.label)
        self._update_label()

    def set_range(self, t_size: int, page_length: int) -> None:
        """Fit the slider to a movie of `t_size` frames; PageUp/PageDown and
        clicking the trough move by one page of `page_length` frames."""
        with QSignalBlocker(self.slider):
            self.slider.setMaximum(max(0, int(t_size) - 1))
            self.slider.setPageStep(max(1, int(page_length)))
        self._update_label()

    def value(self) -> int:
        return int(self.slider.value())

    def set_value_silently(self, timepoint: int) -> None:
        """Move the slider without asking the view to pan (the view moved first)."""
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(timepoint))
        self._update_label()

    def step(self, delta: int) -> None:
        self.slider.setValue(self.slider.value() + int(delta))

    def _on_value_changed(self, value: int) -> None:
        self._update_label()
        self.timepoint_changed.emit(int(value))

    def _update_label(self) -> None:
        self.label.setText(f"t = {self.slider.value()} / {self.slider.maximum()}")
