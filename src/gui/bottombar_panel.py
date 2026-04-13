# from PyQt5.QtWidgets import (
#     QWidget, QPushButton, QSlider, QLabel,
#     QProgressBar, QHBoxLayout, QVBoxLayout,
#     QStackedWidget
# )
# from PyQt5.QtCore import Qt, QTimer, pyqtSignal



# class BottomControlBar(QWidget):
#     frame_changed = pyqtSignal(int)
#     rerender_requested = pyqtSignal()
#     playback_toggled = pyqtSignal(bool)      # True = playing, False = paused

#     MAX_FRAMES = 200
#     TIME_PER_FRAME = 0.1

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._frame = 0

#         # --- Playback controls ---
#         self.play_btn = QPushButton("Play")
#         self.step_back_btn = QPushButton("<-")
#         self.step_fwd_btn = QPushButton("->")

#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setRange(0, self.MAX_FRAMES - 1)

#         self.frame_label = QLabel()
#         self.frame_label.setMinimumWidth(170)

#         self.timer = QTimer(self)
#         self.timer.setInterval(100)

#         # --- Re-render ---
#         self.rerender_btn = QPushButton("Re-render")

#         # --- Info bar ---
#         self.info_stack = QStackedWidget()

#         self.idle_label = QLabel("Ready")
#         self.progress_bar = QProgressBar()
#         self.error_label = QLabel()
#         self.error_label.setStyleSheet("color: red")

#         self.info_stack.addWidget(self.idle_label)    # 0
#         self.info_stack.addWidget(self.progress_bar)  # 1
#         self.info_stack.addWidget(self.error_label)   # 2

#         # --- Layout ---
#         controls = QHBoxLayout()
#         controls.addWidget(self.step_back_btn)
#         controls.addWidget(self.play_btn)
#         controls.addWidget(self.step_fwd_btn)
#         controls.addWidget(self.slider, 1)
#         controls.addWidget(self.frame_label)
#         controls.addWidget(self.rerender_btn)

#         layout = QVBoxLayout(self)
#         layout.addLayout(controls)
#         layout.addWidget(self.info_stack)

#         # --- Signals ---
#         self.play_btn.clicked.connect(self.toggle_play)
#         self.step_back_btn.clicked.connect(self.step_back)
#         self.step_fwd_btn.clicked.connect(self.step_fwd)
#         self.slider.valueChanged.connect(self.set_frame)
#         self.timer.timeout.connect(self.advance_frame)
#         self.rerender_btn.clicked.connect(self.rerender_requested.emit)

#         self._update_label()

#     # ---------------- Playback ----------------

#     def toggle_play(self):
#         if self.timer.isActive():
#             self.timer.stop()
#             self.play_btn.setText("Play")
#             self.playback_toggled.emit(False)
#         else:
#             self.timer.start()
#             self.play_btn.setText("Pause")
#             self.playback_toggled.emit(True)

#     def step_back(self):
#         self.pause()
#         self.set_frame(self._frame - 1)

#     def step_fwd(self):
#         self.pause()
#         self.set_frame(self._frame + 1)

#     def advance_frame(self):
#         self.set_frame(self._frame + 1)

#     def pause(self):
#         self.timer.stop()
#         self.play_btn.setText("Play")
#         self.playback_toggled.emit(False)

#     def set_frame(self, f):
#         f = max(0, min(f, self.MAX_FRAMES - 1))
#         if f == self._frame:
#             return
#         self._frame = f
#         self.slider.blockSignals(True)
#         self.slider.setValue(f)
#         self.slider.blockSignals(False)
#         self._update_label()
#         self.frame_changed.emit(f)

#     def _update_label(self):
#         t = self._frame * self.TIME_PER_FRAME
#         self.frame_label.setText(f"Frame: {self._frame} | Time: {t:.1f}s")

#     # ---------------- Info API ----------------

#     def show_progress(self):
#         self.progress_bar.setValue(0)
#         self.info_stack.setCurrentIndex(1)

#     def set_progress(self, v):
#         self.progress_bar.setValue(v)

#     def show_error(self, msg):
#         self.error_label.setText(msg)
#         self.info_stack.setCurrentIndex(2)

#     def show_ready(self, msg="Ready"):
#         self.idle_label.setText(msg)
#         self.info_stack.setCurrentIndex(0)

#     def set_controls_enabled(self, enabled: bool):
#         for w in (
#             self.play_btn, self.step_back_btn,
#             self.step_fwd_btn, self.slider,
#             self.rerender_btn,
#         ):
#             w.setEnabled(enabled)

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QSlider, QLabel,
    QProgressBar, QHBoxLayout, QVBoxLayout,
    QStackedWidget,
)


class BottomControlBar(QWidget):
    frame_changed = pyqtSignal(int)
    rerender_requested = pyqtSignal()
    playback_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = 0
        self._max_frame = 1
        self._time_per_frame = 0.1

        self.play_btn = QPushButton("Play")
        self.step_back_btn = QPushButton("<-")
        self.step_fwd_btn = QPushButton("->")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setTracking(True)

        self.frame_label = QLabel()
        self.frame_label.setMinimumWidth(170)

        self.timer = QTimer(self)
        self.timer.setInterval(100)

        self.rerender_btn = QPushButton("Re-render")

        self.info_stack = QStackedWidget()
        self.idle_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red")

        self.info_stack.addWidget(self.idle_label)
        self.info_stack.addWidget(self.progress_bar)
        self.info_stack.addWidget(self.error_label)

        controls = QHBoxLayout()
        controls.addWidget(self.step_back_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.step_fwd_btn)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.frame_label)
        controls.addWidget(self.rerender_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.info_stack)

        self.play_btn.clicked.connect(self.toggle_play)
        self.step_back_btn.clicked.connect(self.step_back)
        self.step_fwd_btn.clicked.connect(self.step_fwd)
        self.slider.valueChanged.connect(self.set_frame)
        self.slider.sliderPressed.connect(self.pause)
        self.timer.timeout.connect(self.advance_frame)
        self.rerender_btn.clicked.connect(self.rerender_requested.emit)

        self._update_label()

    def configure_timeline(self, total_frames: int, time_per_frame: float):
        self.pause()
        self._max_frame = max(1, int(total_frames))
        self._time_per_frame = max(0.0, float(time_per_frame))
        self.slider.blockSignals(True)
        self.slider.setRange(0, self._max_frame - 1)
        self.slider.setValue(min(self._frame, self._max_frame - 1))
        self.slider.blockSignals(False)
        if self._frame >= self._max_frame:
            self._frame = self._max_frame - 1
        self._update_label()

    def set_playback_enabled(self, enabled: bool):
        self.play_btn.setEnabled(enabled)
        self.step_back_btn.setEnabled(enabled)
        self.step_fwd_btn.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        if not enabled:
            self.pause()

    def toggle_play(self):
        if self._max_frame <= 1:
            return
        if self.timer.isActive():
            self.pause()
        else:
            if self._frame >= self._max_frame - 1:
                self.set_frame(0, emit_signal=True)
            self.timer.start()
            self.play_btn.setText("Pause")
            self.playback_toggled.emit(True)

    def step_back(self):
        self.pause()
        self.set_frame(self._frame - 1)

    def step_fwd(self):
        self.pause()
        self.set_frame(self._frame + 1)

    def advance_frame(self):
        if self._frame >= self._max_frame - 1:
            self.pause()
            return
        self.set_frame(self._frame + 1)

    def pause(self):
        was_active = self.timer.isActive()
        self.timer.stop()
        self.play_btn.setText("Play")
        if was_active:
            self.playback_toggled.emit(False)

    def set_frame(self, frame_idx, emit_signal: bool = True):
        frame_idx = max(0, min(int(frame_idx), self._max_frame - 1))
        changed = frame_idx != self._frame
        self._frame = frame_idx

        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)

        self._update_label()
        if changed and emit_signal:
            self.frame_changed.emit(frame_idx)

    def _update_label(self):
        t = self._frame * self._time_per_frame
        self.frame_label.setText(f"Frame: {self._frame} | Time: {t:.1f}s")

    def show_progress(self):
        self.progress_bar.setValue(0)
        self.info_stack.setCurrentIndex(1)

    def set_progress(self, value):
        self.progress_bar.setValue(int(value))

    def show_error(self, msg):
        self.error_label.setText(str(msg))
        self.info_stack.setCurrentIndex(2)

    def show_ready(self, msg="Ready"):
        self.idle_label.setText(msg)
        self.info_stack.setCurrentIndex(0)

    def set_controls_enabled(self, enabled: bool):
        self.rerender_btn.setEnabled(enabled)
        if enabled:
            self.set_playback_enabled(self._max_frame > 1)
        else:
            self.pause()
            self.play_btn.setEnabled(False)
            self.step_back_btn.setEnabled(False)
            self.step_fwd_btn.setEnabled(False)
            self.slider.setEnabled(False)

