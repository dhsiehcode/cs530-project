from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import QThread
from gui.sidebar_panel import SidebarPanel
from gui.bottombar_panel import BottomControlBar
from gui.rerender_worker import RerenderWorker
from config import SimConfig



class MainWindow(QMainWindow):
    def __init__(self, config : SimConfig):
        super().__init__()
        self.setWindowTitle("SWE Visualization")
        self.config = config
        # -------- Widgets --------
        self.sidebar = SidebarPanel(config=self.config)
        self.bottom_bar = BottomControlBar()
        self.vtk_widget = QWidget()  # placeholder for QVTKRenderWindowInteractor


        # -------- Layout --------
        center = QWidget()
        main_v = QVBoxLayout(center)

        middle = QHBoxLayout()
        middle.addWidget(self.vtk_widget, 1)
        middle.addWidget(self.sidebar)

        main_v.addLayout(middle, 1)
        main_v.addWidget(self.bottom_bar)

        self.setCentralWidget(center)

        # -------- Signals --------
        self.bottom_bar.rerender_requested.connect(self.start_rerender)
        self.sidebar.obstacle_added.connect(self.validate_obstacle)



    def start_rerender(self):
        self.bottom_bar.set_controls_enabled(False)
        self.bottom_bar.show_progress()

        self.thread = QThread()
        self.worker = RerenderWorker(b_numpy=None)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.bottom_bar.set_progress)
        self.worker.finished.connect(self.on_rerender_finished)
        self.worker.error.connect(self.on_rerender_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_rerender_finished(self):
        self.bottom_bar.show_ready("Re-render complete")
        self.bottom_bar.set_controls_enabled(True)

    def on_rerender_error(self, msg):
        self.bottom_bar.show_error(msg)
        self.bottom_bar.set_controls_enabled(True)

    # --------------------------------------------------
    # Obstacle validation feedback
    # --------------------------------------------------

    def validate_obstacle(self, obstacle, x, y):
        if not self.is_legal_position(obstacle, x, y):
            self.bottom_bar.show_error("Illegal obstacle placement")

    def is_legal_position(self, obstacle, x, y):
        # Your validation logic
        return True
