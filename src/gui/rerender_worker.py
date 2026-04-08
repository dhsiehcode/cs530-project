# render_worker.py

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QSlider, QLabel,
    QProgressBar, QHBoxLayout, QVBoxLayout,
    QStackedWidget
)

class RerenderWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, b_numpy):
        super().__init__()
        self.b_numpy = b_numpy

    def run(self):
        try:
            self.progress.emit(10)

            # b.from_numpy(self.b_numpy)
            self.progress.emit(30)

            # SWE solver
            for i in range(5):
                QThread.msleep(150)
                self.progress.emit(40 + i * 10)

            # Write .vti files
            self.progress.emit(95)

        except Exception as e:
            self.error.emit(str(e))
            return

        self.progress.emit(100)
        self.finished.emit()