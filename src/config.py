import os
from dataclasses import dataclass


@dataclass
class SimConfig:
    nx: int = 256
    ny: int = 128
    dx: float = 0.039
    dy: float = 0.039
    h0: float = 0.1
    ux: float = 0.5
    uy: float = 0.0
    v: float = 0.011
    g: float = 9.81
    dt: float = 0.005
    export_interval: float = 0.1

    # Was 10.0 s, which is too short for particles to cross a ~9.984 m domain
    # at ~0.5 m/s. 24.0 s gives enough time for full downstream advection.
    sim_time: float = 24.0

    use_gpu: bool = False
    warp_scale: float = 4.0
    live_preview_nx: int = 256
    live_preview_ny: int = 128
    live_preview_max_fps: int = 8
    live_preview_range_update_interval: int = 4
    contour_z_offset: float = 0.07
    particle_z_offset: float = 0.14
    particle_trail_z_offset: float = 0.11

    particle_radius: float = 0.030
    particle_trail_radius: float = 0.012
    particle_trail_length: int = 16
    particle_inlet_seed_count: int = 56
    particle_focus_seed_count: int = 72
    particle_focus_half_height: float = 0.85
    particle_focus_upstream_offset: float = 0.55
    particle_respawn_speed_threshold: float = 0.010
    particle_respawn_jump_threshold: float = 0.45

    # Number of grid cells from each y-boundary to exclude from seeding and display
    wall_buffer_cells: int = 0
    # Number of grid cells from the x_max (outlet) boundary to clip from display
    x_max_buffer_cells: int = 45

    x_outlet_buffer_cells: int = 5

    log_buffer_cells: int = 1
    
    @property
    def num_frames(self) -> int:
        return int(self.sim_time / self.export_interval)

    @property
    def steps_per_frame(self) -> int:
        return int(self.export_interval / self.dt)

    @property
    def domain_width(self) -> float:
        return self.nx * self.dx

    @property
    def domain_height(self) -> float:
        return self.ny * self.dy


@dataclass
class ObstacleDef:
    name: str
    kind: str
    radius: float
    height: float
    length: float = 0.0
    angle: float = 0.0


@dataclass
class PlacedObstacle:
    definition: ObstacleDef
    obstacle_id: int
    x: float
    y: float


PRECONFIGURED_OBSTACLES = [
    ObstacleDef("Small Rock", "rock", radius=0.3 * 2, height=0.30),
    ObstacleDef("Medium Rock", "rock", radius=0.4 * 2, height=0.40),
    ObstacleDef("Large Rock", "rock", radius=0.55 * 2, height=0.55),
    ObstacleDef("Small Log", "log", radius=0.15, height=0.40, length=1.5),
    ObstacleDef("Large Log", "log", radius=0.20, height=0.50, length=3.0),
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frames")