from PyQt5.QtCore import QObject, pyqtSignal, QThread

from config import SimConfig, DATA_DIR
from simulation.solver import SWESolver, init_taichi
from simulation.export import export_frame
from simulation.obstacles import build_bed_elevation

class RerenderWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: SimConfig, obstacles: list):
        super().__init__()
        self.config = config
        self.obstacles = obstacles

    def run(self):
        try:
            self.progress.emit(0)
            init_taichi(self.config.use_gpu)
            total_steps = self.config.num_frames * self.config.steps_per_frame
            solver = SWESolver(
                self.config.nx,
                self.config.ny,
                self.config.dx,
                self.config.dy,
                self.config.dt,
                self.config.g,
                self.config.v,
                self.config.h0,
                self.config.ux,
                ramp_steps=total_steps // 10,
            )



            b = build_bed_elevation(self.config, self.obstacles)
            solver.set_bed(b)
            solver.initialize()

            num_frames = self.config.num_frames
            steps_per_frame = self.config.steps_per_frame

            for frame_idx in range(num_frames):
                for _ in range(steps_per_frame):
                    solver.step()
                frame_data = solver.get_frame_data()
                export_frame(frame_data, self.config, frame_idx, DATA_DIR)

                progress = int(((frame_idx + 1) / max(1, num_frames)) * 100)
                self.progress.emit(progress)
        except Exception as e:
            self.error.emit(str(e))
            return

        self.finished.emit()
