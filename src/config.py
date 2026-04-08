import os
from dataclasses import dataclass


@dataclass
class SimConfig:
    nx: int = 512 ## x dimension
    ny: int = 256 ## y dimension
    dx: float = 0.039 ## cell size in grid
    dy: float = 0.039 ## cell size in grid
    h0: float = 0.5 ## surface height
    ux: float = 0.5 ##  velocity in x direction
    uy: float = 0.0 ## velocity in y direction
    v: float = 0.011 ## kinematic viscosity 
    g: float = 9.81 ## gravity
    dt: float = 0.0088 ## Time difference between each solver step  
    export_interval: float = 0.1 ## Frequency of output frame
    sim_time: float = 20.0 ## simulation time 
    use_gpu: bool = False
    warp_scale: float = 3.0
    live_preview_nx: int = 128
    live_preview_ny: int = 64
    live_preview_max_fps: int = 8
    live_preview_range_update_interval: int = 4

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
    kind: str       # "rock" or "log"
    radius: float   # sphere radius (rock) or cross-section radius (log)
    height: float   # peak bed elevation
    length: float = 0.0    # log length (ignored for rocks)
    angle: float = 90.0    # rotation angle in degrees (logs only)


@dataclass
class PlacedObstacle:
    definition: ObstacleDef
    obstacle_id: int
    x: float
    y: float


PRECONFIGURED_OBSTACLES = [
    ObstacleDef("Small Rock",  "rock", radius=0.2,  height=0.20),
    ObstacleDef("Medium Rock", "rock", radius=0.3,  height=0.30),
    ObstacleDef("Large Rock",  "rock", radius=0.5,  height=0.40),
    ObstacleDef("Small Log",   "log",  radius=0.15, height=0.40, length=1.5),
    ObstacleDef("Large Log",   "log",  radius=0.20, height=0.50, length=3.0),
]

# Default output directory for VTI frames
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frames")
