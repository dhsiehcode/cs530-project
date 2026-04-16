import glob
import os

import vtk
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from gui.sidebar_panel import SidebarPanel
from gui.bottombar_panel import BottomControlBar
from gui.rerender_worker import RerenderWorker
from config import SimConfig, DATA_DIR
from visualization.vtk_viz import VTKPipeline


class MainWindow(QMainWindow):
    def __init__(self, config: SimConfig):
        super().__init__()
        self.setWindowTitle("SWE Visualization")
        self.config = config

        self.sidebar = SidebarPanel(config=self.config)
        self.bottom_bar = BottomControlBar()
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

        self.pipeline = VTKPipeline(self.config, self.renderer)
        self.thread = None
        self.worker = None

        center = QWidget()
        main_v = QVBoxLayout(center)
        middle = QHBoxLayout()
        middle.addWidget(self.vtk_widget, 1)
        middle.addWidget(self.sidebar)
        main_v.addLayout(middle, 1)
        main_v.addWidget(self.bottom_bar)
        self.setCentralWidget(center)

        self.bottom_bar.rerender_requested.connect(self.start_rerender)
        self.bottom_bar.frame_changed.connect(self.on_frame_changed)
        self.bottom_bar.playback_toggled.connect(self.on_playback_toggled)
        self.sidebar.obstacle_added.connect(self.validate_obstacle)
        self.sidebar.scalar_field_changed.connect(self.on_scalar_field_changed)
        self.sidebar.layer_toggled.connect(self.on_layer_toggled)
        self.sidebar.obstacles_changed.connect(self.on_obstacles_changed)

        self._init_vtk_scene()

    def _count_saved_frames(self) -> int:
        pattern = os.path.join(DATA_DIR, "frame_*.vti")
        return len(glob.glob(pattern))

    def _init_vtk_scene(self):
        self.bottom_bar.pause()
        saved_frames = self._count_saved_frames()

        if saved_frames > 0:
            skip_frames = max(0, int(round(1.0 / self.config.export_interval)))
            skip_frames = min(skip_frames, saved_frames - 1)
            self.pipeline.load_simulation(
                DATA_DIR,
                saved_frames,
                self.sidebar.placed_obstacles,
            )
            self.bottom_bar.configure_timeline(
                saved_frames, self.config.export_interval, min_frame=skip_frames
            )
            self.bottom_bar.set_playback_enabled(saved_frames > skip_frames + 1)
        else:
            self.pipeline.start_live_mode(
                self.config.live_preview_nx,
                self.config.live_preview_ny,
                self.config.dx,
                self.config.dy,
                self.sidebar.placed_obstacles,
            )
            self.bottom_bar.configure_timeline(1, self.config.export_interval)
            self.bottom_bar.set_playback_enabled(False)

        start_frame = self.bottom_bar._min_frame
        self.pipeline.set_frame(start_frame)
        self.bottom_bar.set_frame(start_frame, emit_signal=False)
        self.pipeline.setup_coordinate_display(self.vtk_widget)
        self.vtk_widget.GetRenderWindow().Render()

    def start_rerender(self):
        self.bottom_bar.pause()
        self.bottom_bar.set_controls_enabled(False)
        self.bottom_bar.show_progress()

        self.thread = QThread()
        self.worker = RerenderWorker(
            config=self.config,
            obstacles=self.sidebar.placed_obstacles,
        )
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
        self._init_vtk_scene()

    def on_rerender_error(self, msg):
        self.bottom_bar.show_error(msg)
        self.bottom_bar.set_controls_enabled(True)

    def validate_obstacle(self, obstacle, x, y):
        if not self.is_legal_position(obstacle, x, y):
            self.bottom_bar.show_error("Illegal obstacle placement")

    def is_legal_position(self, obstacle, x, y):
        return True

    def on_frame_changed(self, idx: int):
        self.pipeline.set_frame(idx)

    def on_playback_toggled(self, playing: bool):
        self.pipeline.set_animating(playing)

    def on_scalar_field_changed(self, label: str):
        mapping = {
            "Height": "h",
            "Speed": "speed",
            "Vorticity": "vorticity",
        }
        field = mapping.get(label)
        if field:
            self.pipeline.set_scalar_field(field)

    def on_layer_toggled(self, layer: str, visible: bool):
        mapping = {
            "water surface": "surface",
            "particles": "particles",
            "particle trails": "particle_trails",
            "contours": "contours",
        }
        target = mapping.get(layer, layer)
        self.pipeline.set_layer_visibility(target, visible)

    def on_obstacles_changed(self):
        self.pipeline.update_obstacles(self.sidebar.placed_obstacles)
