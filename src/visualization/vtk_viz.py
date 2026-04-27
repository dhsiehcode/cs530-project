import math
import os
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from config import SimConfig, PlacedObstacle
from simulation.obstacles import create_obstacle_actor


class VTKPipeline:
    """Manages the VTK rendering pipeline for the SWE visualization."""

    SCALAR_FIELDS = ("h", "speed", "viz_speed", "pressure", "vorticity", "lavd_vorticity")
    SCALAR_LABELS = {
        "h": "Height (m)",
        "speed": "Speed (m/s)",
        "viz_speed": "Velocity (m/s)",
        "pressure": "Pressure",
        "vorticity": "Raw Vorticity (1/s)",
        "lavd_vorticity": "LAVD Vorticity Deviation",
    }

    def __init__(self, config: SimConfig, renderer: vtk.vtkRenderer):
        self.config = config
        self.renderer = renderer
        self.data_dir = ""
        self.num_frames = 0
        self.active_field = "h"

        self.show_surface = True
        self.show_particles = True
        self.show_particle_trails = True
        self.show_contours = True

        self.reader = vtk.vtkXMLImageDataReader()
        self._live_image = None
        self._live_producer = None
        self._live_arrays = {}
        self._live_buffers = {}
        self._live_frame_counter = 0
        self._source = self.reader
        self._is_live = False

        self.surface_actor = None
        self.particle_actor = None
        self.particle_trail_actor = None
        self.contour_actor = None
        self.obstacle_actors: list = []
        self.scalar_bars = {}
        self.corner_annotation = None
        self._coordinate_picker = vtk.vtkWorldPointPicker()
        self._coordinate_interactor = None
        self._mouse_move_observer = None
        self._last_coordinate_text = "X: --  Y: --"

        self._animating = False
        self._obstacles: list[PlacedObstacle] = []
        self._obstacle_flat_mask = None

        self._riverbed_actor = None
        self._water_body_actor = None
        self._surface_mapper = None
        self._particle_mapper = None
        self._particle_trail_mapper = None
        self._contour_mapper = None
        self._particle_poly = None
        self._particle_speed_array = None
        self._particle_trail_poly = None
        self._particle_trail_speed_array = None
        self._particle_seed_pool = np.empty((0, 2), dtype=np.float32)
        self._particle_inlet_seed_count = 0
        self._particle_positions = None
        self._particle_speeds = None
        self._particle_respawns = None
        self._particle_waves: list = []
        self._live_particle_waves: list = []
        self._live_next_spawn_frame = 0
        self._display_voi = None
        

        self._build_color_maps()
        self.scalar_ranges = {
            "h": (0.0, 1.0),
            "speed": (0.0, 1.5),
            "viz_speed": (0.0, 1.5),
            "pressure": (0.0, max(0.10, 0.5 * self.config.g * (1.5 * self.config.h0) ** 2)),
            "vorticity": (-10.0, 10.0),
            "lavd_vorticity": (0.0, 1.0),
        }

    def _get_source_port(self):
        if self._display_voi is not None:
            return self._display_voi.GetOutputPort()
        return self._source.GetOutputPort()

    def _build_display_voi(self):
        data = self._get_current_data()
        if data is None:
            self._display_voi = None
            return
        dims = data.GetDimensions()
        nx, ny = dims[0], dims[1]
        buf = max(1, self.config.wall_buffer_cells)
        x_buf = max(0, self.config.x_max_buffer_cells)
        outleft_buf = max(0, self.config.x_outlet_buffer_cells)
        voi = vtk.vtkExtractVOI()
        voi.SetInputConnection(self._source.GetOutputPort())
        voi.SetVOI(outleft_buf, nx - 1 - x_buf, buf, ny - 1 - buf, 0, 0)
        voi.Update()
        self._display_voi = voi

    def _get_current_data(self):
        return self._live_image if self._is_live else self.reader.GetOutput()

    def _upsert_array(self, point_data, name: str, values: np.ndarray, components: int = 1):
        arr = point_data.GetArray(name)
        vtk_values = np.asarray(values, dtype=np.float32)
        if components > 1:
            vtk_arr = numpy_to_vtk(vtk_values.reshape(-1, components), deep=True)
        else:
            vtk_arr = numpy_to_vtk(vtk_values.reshape(-1), deep=True)
        vtk_arr.SetName(name)

        if arr is None:
            point_data.AddArray(vtk_arr)
            return vtk_arr

        arr.DeepCopy(vtk_arr)
        arr.Modified()
        return arr
    def _bilinear_sample_grid(self, data, field_2d: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Sample a scalar grid at floating-point XY positions."""
        dims = data.GetDimensions()
        origin = data.GetOrigin()
        spacing = data.GetSpacing()
        nx, ny = dims[0], dims[1]

        gx = (positions[:, 0] - origin[0]) / max(spacing[0], 1.0e-8)
        gy = (positions[:, 1] - origin[1]) / max(spacing[1], 1.0e-8)

        gx = np.clip(gx, 0.0, nx - 1.001)
        gy = np.clip(gy, 0.0, ny - 1.001)

        i0 = np.floor(gx).astype(np.int32)
        j0 = np.floor(gy).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, nx - 1)
        j1 = np.clip(j0 + 1, 0, ny - 1)

        tx = (gx - i0).astype(np.float32)
        ty = (gy - j0).astype(np.float32)

        f00 = field_2d[i0, j0]
        f10 = field_2d[i1, j0]
        f01 = field_2d[i0, j1]
        f11 = field_2d[i1, j1]

        return (
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        ).astype(np.float32)


    def _compute_lavd_vorticity_field(self, data) -> np.ndarray:
        """
        Compute a LAVD-style scalar field for the water surface.

        This does not create or modify visible particles.
        It uses hidden massless samples seeded on the grid, advects them through
        the current obstacle-aware velocity field, and accumulates local vorticity
        deviation from the regional mean.
        """
        if data is None:
            return np.empty(0, dtype=np.float32)

        pd = data.GetPointData()
        vx_arr = pd.GetArray("viz_vx")
        vy_arr = pd.GetArray("viz_vy")

        # Fallback to raw velocity if obstacle-aware velocity is unavailable.
        if vx_arr is None or vy_arr is None:
            vx_arr = pd.GetArray("vx")
            vy_arr = pd.GetArray("vy")

        if vx_arr is None or vy_arr is None:
            dims = data.GetDimensions()
            return np.zeros(dims[0] * dims[1], dtype=np.float32)

        dims = data.GetDimensions()
        spacing = data.GetSpacing()
        origin = data.GetOrigin()
        nx, ny = dims[0], dims[1]

        vx = vtk_to_numpy(vx_arr).reshape((nx, ny), order="F").astype(np.float32, copy=False)
        vy = vtk_to_numpy(vy_arr).reshape((nx, ny), order="F").astype(np.float32, copy=False)

        # 2-D vorticity: omega_z = d(vy)/dx - d(vx)/dy
        dvy_dx = np.gradient(vy, max(spacing[0], 1.0e-8), axis=0)
        dvx_dy = np.gradient(vx, max(spacing[1], 1.0e-8), axis=1)
        omega = (dvy_dx - dvx_dy).astype(np.float32)

        # Deviation from regional average, matching the LAVD idea from the slides.
        omega_mean = float(np.mean(omega))
        omega_dev = np.abs(omega - omega_mean).astype(np.float32)

        xs = origin[0] + np.arange(nx, dtype=np.float32) * spacing[0]
        ys = origin[1] + np.arange(ny, dtype=np.float32) * spacing[1]
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        positions = np.column_stack([
            xx.ravel(order="F"),
            yy.ravel(order="F"),
        ]).astype(np.float32)

        accum = np.zeros(positions.shape[0], dtype=np.float32)

        # Hidden massless pathline integration.
        # This is a short-window LAVD proxy over the current frame's velocity field.
        lavd_steps = 10
        dt = float(self.config.export_interval) / float(lavd_steps)

        x_min = origin[0]
        x_max = origin[0] + (nx - 1) * spacing[0]
        y_min = origin[1]
        y_max = origin[1] + (ny - 1) * spacing[1]

        for _ in range(lavd_steps):
            local_dev = self._bilinear_sample_grid(data, omega_dev, positions)
            local_vx = self._bilinear_sample_grid(data, vx, positions)
            local_vy = self._bilinear_sample_grid(data, vy, positions)

            accum += local_dev * dt

            positions[:, 0] += local_vx * dt
            positions[:, 1] += local_vy * dt
            positions[:, 0] = np.clip(positions[:, 0], x_min, x_max)
            positions[:, 1] = np.clip(positions[:, 1], y_min, y_max)

        # Normalize so the surface color map actually shows something.
        hi = float(np.percentile(accum, 98.0)) if accum.size else 1.0
        hi = max(hi, 1.0e-6)
        lavd = np.clip(accum / hi, 0.0, 1.0).astype(np.float32)

        return lavd
    def _apply_obstacle_aware_flow(self):
        data = self._get_current_data()
        if data is None:
            return

        pd = data.GetPointData()
        vx_arr = pd.GetArray("vx")
        vy_arr = pd.GetArray("vy")
        if vx_arr is None or vy_arr is None:
            return

        dims = data.GetDimensions()
        spacing = data.GetSpacing()
        origin = data.GetOrigin()
        nx, ny = dims[0], dims[1]
        if nx <= 0 or ny <= 0:
            return

        xs = origin[0] + np.arange(nx, dtype=np.float32) * spacing[0]
        ys = origin[1] + np.arange(ny, dtype=np.float32) * spacing[1]
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        points = np.column_stack([xx.flatten(order="F"), yy.flatten(order="F")]).astype(np.float32)

        vx = vtk_to_numpy(vx_arr).astype(np.float32, copy=True)
        vy = vtk_to_numpy(vy_arr).astype(np.float32, copy=True)
        vec = np.column_stack([vx, vy]).astype(np.float32, copy=False)

        if self._obstacles:
            shell_pad = max(3.0 * min(self.config.dx, self.config.dy), 0.06)

            def _safe_dir(v: np.ndarray) -> np.ndarray:
                mag = np.linalg.norm(v, axis=1, keepdims=True)
                out = np.zeros_like(v)
                mask = mag[:, 0] > 1.0e-6
                out[mask] = v[mask] / mag[mask]
                out[~mask, 0] = 1.0
                return out

            for obs in self._obstacles:
                defn = obs.definition
                center = np.array([obs.x, obs.y], dtype=np.float32)

                if defn.kind == "rock":
                    closest = np.broadcast_to(center, points.shape)
                    core_radius = max(defn.radius * 0.95, 0.5 * min(self.config.dx, self.config.dy))
                    shell_radius = defn.radius + shell_pad
                else:
                    buf = self.config.log_buffer_cells * self.config.dx
                    angle = np.float32(np.deg2rad(defn.angle + 90))
                    axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                    half_len = 0.5 * float(defn.length) + buf
                    a = center - half_len * axis
                    rel = points - a
                    t = np.clip(rel @ axis, 0.0, float(defn.length) + 2 * buf)
                    closest = a + np.outer(t, axis)
                    core_radius = max(defn.radius * 1.05 + buf, 0.5 * min(self.config.dx, self.config.dy))
                    shell_radius = defn.radius + buf + shell_pad

                delta = points - closest
                dist = np.linalg.norm(delta, axis=1)
                inside_shell = dist < shell_radius
                if not np.any(inside_shell):
                    continue

                normal = np.zeros_like(delta, dtype=np.float32)
                safe_dist = np.maximum(dist, 1.0e-6)
                normal[inside_shell] = delta[inside_shell] / safe_dist[inside_shell, None]

                tangent = np.column_stack([-normal[:, 1], normal[:, 0]]).astype(np.float32, copy=False)
                tangent_flip = np.sum(tangent * vec, axis=1) < 0.0
                tangent[tangent_flip] *= -1.0

                influence = np.clip(
                    (shell_radius - dist) / max(shell_radius - core_radius, 1.0e-6),
                    0.0,
                    1.0,
                ).astype(np.float32)
                core = np.clip(
                    (core_radius - dist) / max(core_radius, 1.0e-6),
                    0.0,
                    1.0,
                ).astype(np.float32)

                inward = np.minimum(np.sum(vec * normal, axis=1), 0.0).astype(np.float32)
                vec = vec - (influence[:, None] * inward[:, None] * normal)

                speed = np.linalg.norm(vec, axis=1).astype(np.float32)
                base_dir = _safe_dir(vec)
                blend = (1.0 - influence)[:, None] * base_dir + influence[:, None] * tangent
                blend = _safe_dir(blend)

                speed *= (1.0 - 0.995 * core)
                vec = blend * speed[:, None]

        viz_speed = np.linalg.norm(vec, axis=1).astype(np.float32)
        viz_velocity = np.column_stack([
            vec[:, 0], vec[:, 1], np.zeros(vec.shape[0], dtype=np.float32)
        ])

        self._upsert_array(pd, "viz_vx", vec[:, 0])
        self._upsert_array(pd, "viz_vy", vec[:, 1])
        self._upsert_array(pd, "viz_speed", viz_speed)
        self._upsert_array(pd, "viz_velocity", viz_velocity, components=3)

        pd.SetActiveScalars("h")
        lavd_vorticity = self._compute_lavd_vorticity_field(data)
        self._upsert_array(pd, "lavd_vorticity", lavd_vorticity)
        pd.SetActiveVectors("velocity")
        data.Modified()

    def load_simulation(self, data_dir: str, num_frames: int, obstacles: list):
        self._is_live = False
        self._source = self.reader
        self.data_dir = data_dir
        self.num_frames = num_frames
        self._obstacles = list(obstacles)

        self._load_frame(0)
        self._estimate_ranges()
        self._precompute_particle_history()
        self._load_frame(0)
        self._refresh_active_arrays()
        self._build_pipeline()
        self._add_obstacles(obstacles)
        self._setup_camera()
        self._update_particle_visuals(0)

    def start_live_mode(self, nx: int, ny: int, dx: float, dy: float, obstacles: list):
        self._is_live = True
        self._live_frame_counter = 0
        self._obstacles = list(obstacles)
        self._particle_positions = None
        self._particle_speeds = None
        self._particle_respawns = None
        self._particle_waves = []
        self._live_particle_waves = []
        self._live_next_spawn_frame = 0
        self._particle_seed_pool = self._build_particle_seed_pool()

        self._live_image = vtk.vtkImageData()
        self._live_image.SetDimensions(nx, ny, 1)
        self._live_image.SetSpacing(dx, dy, 1.0)
        self._live_image.SetOrigin(0, 0, 0)

        pd = self._live_image.GetPointData()
        n = nx * ny
        self._live_arrays = {}
        self._live_buffers = {}
        for name in ("h", "eta", "vx", "vy", "speed", "vorticity", "pressure"):
            buffer = np.full(
                n,
                self.config.h0 if name in ("h", "eta") else 0.0,
                dtype=np.float32,
            )
            arr = numpy_to_vtk(buffer, deep=True)
            arr.SetName(name)
            pd.AddArray(arr)
            self._live_arrays[name] = arr
            self._live_buffers[name] = vtk_to_numpy(arr)

        vec = numpy_to_vtk(np.zeros((n, 3), dtype=np.float32), deep=True)
        vec.SetName("velocity")
        pd.AddArray(vec)
        self._live_arrays["velocity"] = vec
        self._live_buffers["velocity"] = vtk_to_numpy(vec)
        pd.SetActiveScalars("h")
        pd.SetActiveVectors("velocity")

        self._apply_obstacle_aware_flow()

        self._live_producer = vtk.vtkTrivialProducer()
        self._live_producer.SetOutput(self._live_image)
        self._source = self._live_producer

        self.scalar_ranges = {
            "h": (0.0, 1.0),
            "speed": (0.0, 2.0),
            "viz_speed": (0.0, 2.0),
            "pressure": (0.0, max(0.10, 0.5 * self.config.g * (1.5 * self.config.h0) ** 2)),
            "vorticity": (-10.0, 10.0),
            "lavd_vorticity": (0.0, 1.0),
        }

        self._compute_obstacle_grid_mask()
        self._build_pipeline()
        self._add_obstacles(obstacles)
        self._setup_camera()
        self._update_particle_visuals(0)

    def update_live_frame(self, frame_data: dict, render: bool = True):
        if self._live_image is None:
            return

        for name in ("h", "eta", "vx", "vy", "speed", "vorticity", "pressure"):
            flat = frame_data[name].flatten(order="F").astype(np.float32, copy=False)
            if name == "eta" and self._obstacle_flat_mask is not None:
                flat = flat.copy()
                flat[self._obstacle_flat_mask] = self.config.h0
            np.copyto(self._live_buffers[name], flat)
            self._live_arrays[name].Modified()

        vx = frame_data["vx"].flatten(order="F").astype(np.float32)
        vy = frame_data["vy"].flatten(order="F").astype(np.float32)
        velocity = self._live_buffers["velocity"]
        velocity[:, 0] = vx
        velocity[:, 1] = vy
        velocity[:, 2] = 0.0
        self._live_arrays["velocity"].Modified()

        self._refresh_active_arrays()
        pd = self._live_image.GetPointData()

        self._live_frame_counter += 1
        if self._live_frame_counter % max(1, self.config.live_preview_range_update_interval) == 0:
            for name in self.SCALAR_FIELDS:
                arr = pd.GetArray(name)
                if arr:
                    lo_d, hi_d = arr.GetRange()
                    lo_c, hi_c = self.scalar_ranges[name]
                    if name == "vorticity":
                        mx = max(abs(lo_d), abs(hi_d), abs(lo_c), abs(hi_c), 0.1)
                        self.scalar_ranges[name] = (-mx, mx)
                    else:
                        self.scalar_ranges[name] = (min(lo_c, lo_d), max(hi_c, hi_d))
            if self._surface_mapper:
                lo, hi = self.scalar_ranges[self.active_field]
                self._surface_mapper.SetScalarRange(lo, hi)
            self._sync_ranges()

        self._advect_live_particle_waves()
        self._update_particle_visuals(0)
        if render:
            self.renderer.GetRenderWindow().Render()

    def stop_live_mode(self):
        self._is_live = False
        self._live_image = None
        self._live_producer = None
        self._live_arrays = {}
        self._live_buffers = {}
        self._live_frame_counter = 0
        self._source = self.reader
        self.clear()

    def set_animating(self, playing: bool):
        self._animating = playing
        self._update_scalar_bar_visibility()
        self.renderer.GetRenderWindow().Render()

    def set_frame(self, idx: int):
        if not self._load_frame(idx):
            return
        self._refresh_active_arrays()
        self._update_particle_visuals(idx)
        self.renderer.GetRenderWindow().Render()

    def set_scalar_field(self, field_name: str):
        if field_name not in self.SCALAR_FIELDS or not self._surface_mapper:
            return
        self.active_field = field_name
        lo, hi = self.scalar_ranges[field_name]
        self._surface_mapper.SelectColorArray(field_name)
        self._surface_mapper.SetScalarRange(lo, hi)
        self._surface_mapper.SetLookupTable(self.ctfs[field_name])
        self._surface_mapper.SetScalarRange(lo, hi)
        if self.scalar_bars.get("surface"):
            self.scalar_bars["surface"].SetLookupTable(self.ctfs[field_name])
            self.scalar_bars["surface"].SetTitle(self.SCALAR_LABELS[field_name])
        self.renderer.GetRenderWindow().Render()

    def set_layer_visibility(self, layer: str, visible: bool):
        actors = {
            "surface": self.surface_actor,
            "particles": self.particle_actor,
            "particle_trails": self.particle_trail_actor,
            "contours": self.contour_actor,
        }
        actor = actors.get(layer)
        if actor is not None:
            actor.SetVisibility(visible)
            if layer == "surface":
                self.show_surface = visible
            elif layer == "particles":
                self.show_particles = visible
            elif layer == "particle_trails":
                self.show_particle_trails = visible
            elif layer == "contours":
                self.show_contours = visible
            self._update_scalar_bar_visibility()
            self.renderer.GetRenderWindow().Render()

    def setup_coordinate_display(self, interactor):
        if self.corner_annotation is None:
            self.corner_annotation = vtk.vtkCornerAnnotation()
            self.corner_annotation.SetText(2, self._last_coordinate_text)
            self.corner_annotation.GetTextProperty().SetFontSize(14)
            self.corner_annotation.GetTextProperty().SetColor(1, 1, 1)
            self.renderer.AddViewProp(self.corner_annotation)

        if self._coordinate_interactor is interactor and self._mouse_move_observer is not None:
            return

        if self._coordinate_interactor and self._mouse_move_observer is not None:
            self._coordinate_interactor.RemoveObserver(self._mouse_move_observer)

        def _on_mouse_move(_obj, _event):
            x, y = interactor.GetEventPosition()
            self._coordinate_picker.Pick(x, y, 0, self.renderer)
            pos = self._coordinate_picker.GetPickPosition()
            text = f"X: {pos[0]:.2f} m   Y: {pos[1]:.2f} m"
            if text == self._last_coordinate_text:
                return
            self._last_coordinate_text = text
            self.corner_annotation.SetText(2, text)
            interactor.GetRenderWindow().Render()

        self._coordinate_interactor = interactor
        self._mouse_move_observer = interactor.AddObserver("MouseMoveEvent", _on_mouse_move)

    def clear(self):
        self.renderer.RemoveAllViewProps()
        self.surface_actor = None
        self.particle_actor = None
        self.particle_trail_actor = None
        self.contour_actor = None
        self._riverbed_actor = None
        self._water_body_actor = None
        self._surface_mapper = None
        self._particle_mapper = None
        self._particle_trail_mapper = None
        self._contour_mapper = None
        self.obstacle_actors.clear()
        self.scalar_bars = {}
        self.corner_annotation = None
        self._particle_poly = None
        self._particle_speed_array = None
        self._particle_trail_poly = None
        self._particle_trail_speed_array = None
        self._display_voi = None

    def update_obstacles(self, obstacles: list):
        self._obstacles = list(obstacles)
        current_frame = 0
        if not self._is_live:
            file_name = self.reader.GetFileName() or ""
            try:
                current_frame = int(os.path.basename(file_name).split("_")[1].split(".")[0])
            except Exception:
                current_frame = 0
            self._precompute_particle_history()
            self._load_frame(current_frame)
            self._refresh_active_arrays()
        else:
            self._particle_seed_pool = self._build_particle_seed_pool()

        for actor in self.obstacle_actors:
            self.renderer.RemoveActor(actor)
        self.obstacle_actors.clear()
        self._add_obstacles(obstacles)
        self._apply_obstacle_aware_flow()
        self._update_particle_visuals(current_frame)
        self.renderer.GetRenderWindow().Render()

    def _load_frame(self, idx: int) -> bool:
        path = os.path.join(self.data_dir, f"frame_{idx:04d}.vti")
        if not os.path.exists(path):
            return False
        self.reader.SetFileName(path)
        self.reader.Update()
        self._refresh_active_arrays()
        return True

    def _refresh_active_arrays(self):
        data = self.reader.GetOutput() if not self._is_live else self._live_image
        if data:
            data.GetPointData().SetActiveScalars("h")
            data.GetPointData().SetActiveVectors("velocity")
            self._apply_obstacle_aware_flow()

    def _estimate_ranges(self):
        data = self._live_image if self._is_live else self.reader.GetOutput()
        if not data:
            return
        pd = data.GetPointData()
        for name in self.SCALAR_FIELDS:
            arr = pd.GetArray(name)
            if arr:
                lo, hi = arr.GetRange()
                if name == "vorticity":
                    mx = max(abs(lo), abs(hi), 0.1)
                    lo, hi = -mx, mx
                elif lo == hi:
                    hi = lo + 1.0
                self.scalar_ranges[name] = (lo, hi)
        self._sync_ranges()

    def _sync_ranges(self):
        lo, hi = self.scalar_ranges["speed"]
        if self._particle_mapper:
            self._particle_mapper.SetScalarRange(lo, hi)
        if self._particle_trail_mapper:
            self._particle_trail_mapper.SetScalarRange(lo, hi)
        if self._contour_mapper:
            vlo, vhi = self.scalar_ranges["vorticity"]
            self._contour_mapper.SetScalarRange(vlo, vhi)

    def _build_color_maps(self):
        self.ctfs = {}

        ctf_h = vtk.vtkColorTransferFunction()
        ctf_h.SetColorSpaceToLab()
        ctf_h.AddRGBPoint(0.0, 0.02, 0.08, 0.30)
        ctf_h.AddRGBPoint(0.25, 0.04, 0.18, 0.52)
        ctf_h.AddRGBPoint(0.50, 0.06, 0.38, 0.66)
        ctf_h.AddRGBPoint(0.70, 0.10, 0.58, 0.74)
        ctf_h.AddRGBPoint(0.85, 0.25, 0.72, 0.82)
        ctf_h.AddRGBPoint(1.0, 0.55, 0.88, 0.94)
        self.ctfs["h"] = ctf_h

        ctf_s = vtk.vtkColorTransferFunction()
        ctf_s.SetColorSpaceToLab()
        ctf_s.AddRGBPoint(0.0,  1.00, 1.00, 0.00)
        ctf_s.AddRGBPoint(0.33, 1.00, 0.75, 0.00)
        ctf_s.AddRGBPoint(0.66, 1.00, 0.40, 0.00)
        ctf_s.AddRGBPoint(1.0,  0.85, 0.05, 0.05)
        self.ctfs["speed"] = ctf_s
        self.ctfs["viz_speed"] = ctf_s

        ctf_p = vtk.vtkColorTransferFunction()
        ctf_p.SetColorSpaceToLab()
        ctf_p.AddRGBPoint(0.0, 0.07, 0.10, 0.28)
        ctf_p.AddRGBPoint(0.20, 0.10, 0.32, 0.62)
        ctf_p.AddRGBPoint(0.45, 0.26, 0.62, 0.74)
        ctf_p.AddRGBPoint(0.70, 0.86, 0.74, 0.26)
        ctf_p.AddRGBPoint(1.0, 0.78, 0.16, 0.08)
        self.ctfs["pressure"] = ctf_p

        ctf_v = vtk.vtkColorTransferFunction()
        ctf_v.SetColorSpaceToDiverging()
        ctf_v.AddRGBPoint(-10.0, 0.231, 0.298, 0.753)
        ctf_v.AddRGBPoint(0.0, 0.865, 0.865, 0.865)
        ctf_v.AddRGBPoint(10.0, 0.706, 0.016, 0.150)
        self.ctfs["vorticity"] = ctf_v


        ctf_lavd = vtk.vtkColorTransferFunction()
        ctf_lavd.SetColorSpaceToLab()
        ctf_lavd.AddRGBPoint(0.0, 0.02, 0.05, 0.18)
        ctf_lavd.AddRGBPoint(0.20, 0.04, 0.22, 0.45)
        ctf_lavd.AddRGBPoint(0.45, 0.00, 0.55, 0.70)
        ctf_lavd.AddRGBPoint(0.70, 0.95, 0.75, 0.20)
        ctf_lavd.AddRGBPoint(1.0, 0.90, 0.10, 0.05)
        self.ctfs["lavd_vorticity"] = ctf_lavd

    def _build_pipeline(self):
        self.clear()
        self._build_display_voi()
        self._build_riverbed()
        self._build_water_body()
        self._build_surface()
        self._build_contours()
        self._build_particles()
        self._build_particle_trails()
        self._build_scalar_bars()
        self._setup_lighting()

    def _build_riverbed(self):
        w = self.config.domain_width
        h = self.config.domain_height
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(self.config.x_outlet_buffer_cells * self.config.dx, 0, -0.01)
        plane.SetPoint1(w - self.config.x_max_buffer_cells * self.config.dx, 0, -0.01)
        plane.SetPoint2(self.config.x_outlet_buffer_cells * self.config.dx, h, -0.01)
        plane.SetXResolution(1)
        plane.SetYResolution(1)
        plane.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.58, 0.50, 0.36)
        actor.GetProperty().SetAmbient(0.4)
        actor.GetProperty().SetDiffuse(0.6)

        self.renderer.AddActor(actor)
        self._riverbed_actor = actor

    def _build_water_body(self):
        """Solid enclosed cube representing the water volume below the surface."""
        x0 = self.config.x_outlet_buffer_cells * self.config.dx
        x1 = self.config.domain_width - self.config.x_max_buffer_cells * self.config.dx
        y_max = self.config.domain_height

        # Keep the water body just below the mean free surface so the ripple reads clearly.
        wall_h = max(0.0, self.config.warp_scale * self.config.h0 - 0.02)

        cube = vtk.vtkCubeSource()
        cube.SetBounds(x0, x1, 0.0, y_max, 0.0, wall_h)
        cube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.06, 0.38, 0.66)
        prop.SetOpacity(0.25)
        prop.SetSpecular(0.20)
        prop.SetSpecularPower(25)
        prop.SetSpecularColor(1.0, 1.0, 1.0)
        prop.SetDiffuse(0.7)
        prop.SetAmbient(0.16)

        self.renderer.AddActor(actor)
        self._water_body_actor = actor

    def _surface_offset_filter(self, input_port):
        geom = vtk.vtkImageDataGeometryFilter()
        geom.SetInputConnection(input_port)

        calc = vtk.vtkArrayCalculator()
        calc.SetInputConnection(geom.GetOutputPort())
        calc.SetAttributeTypeToPointData()
        calc.AddScalarArrayName("eta", 0)
        calc.SetResultArrayName("surface_offset")

        # Warp only the deviation from the rest water level, not the full absolute height.
        calc.SetFunction(f"(eta - {self.config.h0})")

        assign = vtk.vtkAssignAttribute()
        assign.SetInputConnection(calc.GetOutputPort())
        assign.Assign(
            "surface_offset",
            vtk.vtkDataSetAttributes.SCALARS,
            vtk.vtkAssignAttribute.POINT_DATA,
        )
        return assign

    def _build_surface(self):
        assign = self._surface_offset_filter(self._get_source_port())

        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(assign.GetOutputPort())

        # Exaggerate only the wave deviation.
        # warp.SetScaleFactor(self.config.warp_scale * 6.0)
        warp.SetScaleFactor(self.config.warp_scale * 3.0)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(warp.GetOutputPort())
        normals.SetFeatureAngle(60.0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(self.active_field)
        lo, hi = self.scalar_ranges[self.active_field]
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs[self.active_field])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Place the surface at the mean water level first, then add the amplified ripple.
        actor.AddPosition(0, 0, self.config.warp_scale * self.config.h0 + 0.02)

        prop = actor.GetProperty()
        prop.SetOpacity(0.92)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(40)
        prop.SetSpecularColor(1.0, 1.0, 1.0)
        prop.SetDiffuse(0.7)
        prop.SetAmbient(0.16)
        actor.SetVisibility(self.show_surface)

        self.renderer.AddActor(actor)
        self.surface_actor = actor
        self._surface_mapper = mapper

    def _build_particles(self):
        self._particle_poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(0)
        self._particle_poly.SetPoints(points)

        verts = vtk.vtkCellArray()
        self._particle_poly.SetVerts(verts)

        self._particle_speed_array = vtk.vtkFloatArray()
        self._particle_speed_array.SetName("particle_speed")
        self._particle_poly.GetPointData().AddArray(self._particle_speed_array)
        self._particle_poly.GetPointData().SetActiveScalars("particle_speed")

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(self.config.particle_radius)
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(12)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self._particle_poly)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.ScalingOff()
        glyph.OrientOff()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("particle_speed")
        lo, hi = self.scalar_ranges["speed"]
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["speed"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        base_z = self.config.warp_scale * self.config.h0
        actor.AddPosition(0, 0, base_z + self.config.particle_z_offset)
        actor.GetProperty().SetOpacity(0.95)
        actor.SetVisibility(self.show_particles)

        self.renderer.AddActor(actor)
        self.particle_actor = actor
        self._particle_mapper = mapper

    def _build_particle_trails(self):
        self._particle_trail_poly = vtk.vtkPolyData()
        self._particle_trail_poly.SetPoints(vtk.vtkPoints())
        self._particle_trail_poly.SetLines(vtk.vtkCellArray())

        self._particle_trail_speed_array = vtk.vtkFloatArray()
        self._particle_trail_speed_array.SetName("particle_speed")
        self._particle_trail_poly.GetPointData().AddArray(self._particle_trail_speed_array)
        self._particle_trail_poly.GetPointData().SetActiveScalars("particle_speed")

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(self._particle_trail_poly)
        tube.SetRadius(self.config.particle_trail_radius)
        tube.SetNumberOfSides(10)
        tube.CappingOff()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("particle_speed")
        lo, hi = self.scalar_ranges["speed"]
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["speed"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        base_z = self.config.warp_scale * self.config.h0
        actor.AddPosition(0, 0, base_z + self.config.particle_trail_z_offset)
        actor.GetProperty().SetOpacity(0.90)
        actor.SetVisibility(self.show_particle_trails)

        self.renderer.AddActor(actor)
        self.particle_trail_actor = actor
        self._particle_trail_mapper = mapper

    def _obstacle_local_contour_levels(self, data, global_mag: float, n_local: int = 6) -> np.ndarray:
        """Return extra contour levels densely packed in the high-vorticity zones near each obstacle."""
        if not self._obstacles or data is None:
            return np.empty(0, dtype=np.float32)

        arr = data.GetPointData().GetArray("vorticity")
        if arr is None:
            return np.empty(0, dtype=np.float32)

        dims = data.GetDimensions()
        nx, ny = dims[0], dims[1]
        origin = data.GetOrigin()
        spacing = data.GetSpacing()

        xs = origin[0] + np.arange(nx, dtype=np.float32) * spacing[0]
        ys = origin[1] + np.arange(ny, dtype=np.float32) * spacing[1]
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([xx.ravel(order="F"), yy.ravel(order="F")])
        vort = vtk_to_numpy(arr).astype(np.float32)

        extra = []
        for obs in self._obstacles:
            defn = obs.definition
            center = np.array([obs.x, obs.y], dtype=np.float32)

            if defn.kind == "rock":
                search_r = defn.radius * 4.0
                dist = np.linalg.norm(pts - center, axis=1)
                mask = dist <= search_r
            else:
                angle = np.deg2rad(defn.angle)
                axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                half_len = 0.5 * defn.length
                a = center - half_len * axis
                rel = pts - a
                t = np.clip(rel @ axis, 0.0, defn.length)
                closest = a + np.outer(t, axis)
                dist = np.linalg.norm(pts - closest, axis=1)
                search_r = defn.radius * 4.0 + 0.5 * defn.length + 0.5 * SimConfig.log_buffer_cells * SimConfig.dx
                mask = dist <= search_r

            if mask.sum() < 4:
                continue

            local_vort = vort[mask]
            local_mag = float(np.percentile(np.abs(local_vort), 90))
            local_mag = max(local_mag, global_mag * 0.3)

            # Only add levels that are denser than what the global levels already provide
            if local_mag <= global_mag * 1.1:
                continue

            for sign in (-1.0, 1.0):
                levels = np.linspace(sign * global_mag * 0.25, sign * local_mag, n_local,
                                     dtype=np.float32)
                extra.append(levels)

        if not extra:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(extra)

    def _build_contours(self):
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(self._get_source_port())
        contour.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vorticity"
        )

        lo, hi = self.scalar_ranges["vorticity"]
        global_mag = max(abs(lo), abs(hi), 0.1)

        data = self._get_current_data()
        if data is not None:
            arr = data.GetPointData().GetArray("vorticity")
            if arr is not None:
                values = vtk_to_numpy(arr).astype(np.float32)
                nonzero = np.abs(values[np.abs(values) > 1.0e-6])
                if nonzero.size:
                    global_mag = float(np.percentile(nonzero, 95))
                    global_mag = max(global_mag, 0.1)

        # Keep contours simple and global only.
        # The old code added obstacle-local contour levels, which made the tubes
        # bunch up into a jagged ring around rocks.
        pos = np.array([0.12, 0.22, 0.36, 0.55], dtype=np.float32) * global_mag
        levels = np.concatenate([-pos[::-1], pos])

        contour.SetNumberOfContours(len(levels))
        for idx, value in enumerate(levels):
            contour.SetValue(idx, float(value))

        base_z = self.config.warp_scale * self.config.h0
        transform = vtk.vtkTransform()
        transform.Translate(0, 0, base_z + self.config.contour_z_offset)

        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(contour.GetOutputPort())
        tf.SetTransform(transform)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.014 if self._is_live else 0.009)
        tube.SetNumberOfSides(12)
        tube.CappingOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("vorticity")
        mapper.SetScalarRange(-global_mag, global_mag)
        mapper.SetLookupTable(self.ctfs["vorticity"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.80)
        actor.SetVisibility(self.show_contours)

        self.renderer.AddActor(actor)
        self.contour_actor = actor
        self._contour_mapper = mapper

    def _make_scalar_bar(self, title, lookup_table, x, y, h):
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lookup_table)
        bar.SetTitle(title)
        bar.SetNumberOfLabels(4)
        bar.SetWidth(0.07)
        bar.SetHeight(h)
        bar.SetPosition(x, y)
        bar.GetTitleTextProperty().SetFontSize(11)
        bar.GetTitleTextProperty().SetColor(1, 1, 1)
        bar.GetLabelTextProperty().SetColor(1, 1, 1)
        return bar

    def _build_scalar_bars(self):
        surface_bar = self._make_scalar_bar(
            self.SCALAR_LABELS[self.active_field],
            self.ctfs[self.active_field],
            0.91,
            0.05,
            0.28,
        )
        particle_bar = self._make_scalar_bar("Particle Speed", self.ctfs["speed"], 0.01, 0.67, 0.24)
        trail_bar = self._make_scalar_bar("Trail Speed", self.ctfs["speed"], 0.01, 0.38, 0.24)
        contour_bar = self._make_scalar_bar("Vorticity", self.ctfs["vorticity"], 0.01, 0.09, 0.24)

        for bar in (surface_bar, particle_bar, trail_bar, contour_bar):
            self.renderer.AddActor2D(bar)

        self.scalar_bars = {
            "surface": surface_bar,
            "particles": particle_bar,
            "particle_trails": trail_bar,
            "contours": contour_bar,
        }
        self._update_scalar_bar_visibility()

    def _update_scalar_bar_visibility(self):
        if not self.scalar_bars:
            return
        self.scalar_bars["surface"].SetVisibility(self.show_surface)
        self.scalar_bars["particles"].SetVisibility(self.show_particles)
        self.scalar_bars["particle_trails"].SetVisibility(self.show_particle_trails)
        self.scalar_bars["contours"].SetVisibility(self.show_contours)

    def _compute_obstacle_grid_mask(self):
        """Precompute a flat boolean mask of obstacle footprints on the live grid."""
        self._obstacle_flat_mask = None
        if not self._obstacles or self._live_image is None:
            return
        dims = self._live_image.GetDimensions()
        nx, ny = dims[0], dims[1]
        orig = self._live_image.GetOrigin()
        sp = self._live_image.GetSpacing()
        xs = orig[0] + np.arange(nx, dtype=np.float32) * sp[0]
        ys = orig[1] + np.arange(ny, dtype=np.float32) * sp[1]
        ii, jj = np.meshgrid(xs, ys, indexing="ij")
        combined = np.zeros((nx, ny), dtype=bool)
        for obs in self._obstacles:
            defn = obs.definition
            cx, cy = obs.x, obs.y
            if defn.kind == "rock":
                dist = np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)
                combined |= dist < defn.radius
            elif defn.kind == "log":
                buf = self.config.log_buffer_cells * self.config.dx
                angle_rad = math.radians(defn.angle + 90)
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                dx_local = (ii - cx) * cos_a + (jj - cy) * sin_a
                dy_local = -(ii - cx) * sin_a + (jj - cy) * cos_a
                combined |= (np.abs(dx_local) < defn.length / 2 + buf) & (np.abs(dy_local) < defn.radius + buf)
        self._obstacle_flat_mask = combined.flatten(order="F")

    def _point_in_obstacle(self, x: float, y: float) -> bool:
        """Return True if world point (x, y) lies inside any obstacle's footprint."""
        for obs in self._obstacles:
            defn = obs.definition
            cx, cy = obs.x, obs.y
            if defn.kind == "rock":
                if (x - cx) ** 2 + (y - cy) ** 2 < defn.radius ** 2:
                    return True
            elif defn.kind == "log":
                buf = self.config.log_buffer_cells * self.config.dx
                angle_rad = math.radians(defn.angle + 90)
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                dx_local = (x - cx) * cos_a + (y - cy) * sin_a
                dy_local = -(x - cx) * sin_a + (y - cy) * cos_a
                if abs(dx_local) < defn.length / 2 + buf and abs(dy_local) < defn.radius + buf:
                    return True
        return False

    def _add_obstacles(self, obstacles: list):
        for obs in obstacles:
            actor = create_obstacle_actor(obs, self.config.warp_scale)
            self.renderer.AddActor(actor)
            self.obstacle_actors.append(actor)

    def _setup_lighting(self):
        self.renderer.RemoveAllLights()

        key = vtk.vtkLight()
        key.SetLightTypeToSceneLight()
        w = self.config.domain_width
        h = self.config.domain_height
        key.SetPosition(w * 0.3, -h * 0.5, h * 2.0)
        key.SetFocalPoint(w / 2, h / 2, 0)
        key.SetColor(1.0, 0.98, 0.94)
        key.SetIntensity(1.0)
        self.renderer.AddLight(key)

        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(w * 0.8, h * 1.5, h * 0.8)
        fill.SetFocalPoint(w / 2, h / 2, 0)
        fill.SetColor(0.85, 0.90, 1.0)
        fill.SetIntensity(0.4)
        self.renderer.AddLight(fill)

    def _setup_camera(self):
        w = self.config.domain_width
        h = self.config.domain_height
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(w * 0.28, -h * 0.78, h * 0.92)
        cam.SetFocalPoint(w * 0.55, h / 2, 0.10)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        self.renderer.SetBackground(0.12, 0.14, 0.22)
        self.renderer.SetBackground2(0.28, 0.35, 0.50)
        self.renderer.GradientBackgroundOn()

    def _advect_live_particle_waves(self):
        data = self._live_image
        if data is None or self._particle_seed_pool.size == 0:
            return

        spawn_interval = max(1, round(self.config.particle_interval / self.config.export_interval))
        if (self._live_frame_counter - 1) % spawn_interval == 0:
            self._live_particle_waves.append({
                'positions': self._particle_seed_pool.copy(),
                'speeds': np.zeros(self._particle_seed_pool.shape[0], dtype=np.float32),
            })

        dt = self.config.export_interval
        x_min = self.config.dx
        x_max = self.config.domain_width - self.config.dx
        wall_buf = self.config.wall_buffer_cells * self.config.dy
        y_min = wall_buf
        y_max = self.config.domain_height - wall_buf

        for wave in self._live_particle_waves:
            positions = wave['positions']
            vel = self._sample_velocity(data, positions)
            wave['speeds'] = np.linalg.norm(vel, axis=1).astype(np.float32)
            new_pos = (positions + dt * vel).astype(np.float32)
            new_pos[:, 0] = np.clip(new_pos[:, 0], x_min, x_max)
            new_pos[:, 1] = np.clip(new_pos[:, 1], y_min, y_max)
            wave['positions'] = new_pos

    def _build_particle_seed_pool(self) -> np.ndarray:
        x_inlet = self.config.dx * 1.5 + self.config.x_outlet_buffer_cells * self.config.dx
        buf = self.config.wall_buffer_cells * self.config.dy
        y_min = buf
        y_max = self.config.domain_height - buf

        seeds = []
        inlet_count = max(8, int(self.config.particle_inlet_seed_count))
        for y in np.linspace(y_min, y_max, inlet_count, dtype=np.float32):
            seeds.append([x_inlet, y])
        self._particle_inlet_seed_count = inlet_count

        if not seeds:
            return np.empty((0, 2), dtype=np.float32)

        seed_pool = np.asarray(seeds, dtype=np.float32)
        seed_pool[:, 0] = np.clip(seed_pool[:, 0], x_inlet, self.config.domain_width - self.config.dx * 2.0)
        seed_pool[:, 1] = np.clip(seed_pool[:, 1], y_min, y_max)
        return seed_pool

    def _points_inside_obstacles(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or not self._obstacles:
            return np.zeros(points.shape[0], dtype=bool)

        inside = np.zeros(points.shape[0], dtype=bool)
        for obs in self._obstacles:
            defn = obs.definition
            center = np.array([obs.x, obs.y], dtype=np.float32)
            if defn.kind == "rock":
                dist = np.linalg.norm(points - center, axis=1)
                inside |= dist <= max(defn.radius * 0.95, min(self.config.dx, self.config.dy))
            else:
                buf = self.config.log_buffer_cells * self.config.dx
                angle = np.deg2rad(defn.angle + 90)
                axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                half_len = 0.5 * defn.length + buf
                a = center - half_len * axis
                rel = points - a
                t = np.clip(rel @ axis, 0.0, defn.length + 2 * buf)
                closest = a + np.outer(t, axis)
                dist = np.linalg.norm(points - closest, axis=1)
                inside |= dist <= max(defn.radius + buf, min(self.config.dx, self.config.dy))
        return inside

    def _sample_velocity(self, data, positions: np.ndarray) -> np.ndarray:
        if positions.size == 0 or data is None:
            return np.zeros((0, 2), dtype=np.float32)

        pd = data.GetPointData()
        vx_arr = pd.GetArray("viz_vx") or pd.GetArray("vx")
        vy_arr = pd.GetArray("viz_vy") or pd.GetArray("vy")
        if vx_arr is None or vy_arr is None:
            return np.zeros((positions.shape[0], 2), dtype=np.float32)

        dims = data.GetDimensions()
        nx, ny = dims[0], dims[1]
        if nx < 2 or ny < 2:
            return np.zeros((positions.shape[0], 2), dtype=np.float32)

        origin = data.GetOrigin()
        spacing = data.GetSpacing()
        fx = (positions[:, 0] - origin[0]) / max(spacing[0], 1.0e-6)
        fy = (positions[:, 1] - origin[1]) / max(spacing[1], 1.0e-6)

        fx = np.clip(fx, 0.0, nx - 1.001)
        fy = np.clip(fy, 0.0, ny - 1.001)

        i0 = np.floor(fx).astype(np.int32)
        j0 = np.floor(fy).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, nx - 1)
        j1 = np.clip(j0 + 1, 0, ny - 1)
        tx = (fx - i0).astype(np.float32)
        ty = (fy - j0).astype(np.float32)

        vx = vtk_to_numpy(vx_arr).reshape((nx, ny), order="F").astype(np.float32, copy=False)
        vy = vtk_to_numpy(vy_arr).reshape((nx, ny), order="F").astype(np.float32, copy=False)

        def interp(field):
            f00 = field[i0, j0]
            f10 = field[i1, j0]
            f01 = field[i0, j1]
            f11 = field[i1, j1]
            return (
                (1.0 - tx) * (1.0 - ty) * f00
                + tx * (1.0 - ty) * f10
                + (1.0 - tx) * ty * f01
                + tx * ty * f11
            )

        return np.column_stack([interp(vx), interp(vy)]).astype(np.float32)

    def _precompute_particle_history(self):
        self._particle_seed_pool = self._build_particle_seed_pool()
        self._particle_waves = []
        if self._particle_seed_pool.size == 0 or self.num_frames <= 0:
            self._particle_positions = np.zeros((0, 0, 2), dtype=np.float32)
            self._particle_speeds = np.zeros((0, 0), dtype=np.float32)
            self._particle_respawns = np.zeros((0, 0), dtype=bool)
            return

        n_particles = self._particle_seed_pool.shape[0]
        rng = np.random.default_rng(20260414)
        spawn_interval = max(1, round(self.config.particle_interval / self.config.export_interval))

        dt = float(self.config.export_interval)
        x_min = self.config.dx * 1.0
        x_max = self.config.domain_width - self.config.dx * 1.0
        buf = self.config.wall_buffer_cells * self.config.dy
        y_min = buf
        y_max = self.config.domain_height - buf

        active_waves = []

        for frame_idx in range(self.num_frames):
            if not self._load_frame(frame_idx):
                break
            data = self.reader.GetOutput()

            if frame_idx % spawn_interval == 0:
                active_waves.append({
                    'spawn_frame': frame_idx,
                    'positions': self._particle_seed_pool.copy(),
                    'history': [],
                    'speeds': [],
                    'respawns': [],
                    'pending_respawn_mask': np.zeros(n_particles, dtype=bool),
                })

            for wave in active_waves:
                positions = wave['positions']

                wave['respawns'].append(wave['pending_respawn_mask'].copy())
                wave['pending_respawn_mask'][:] = False

                wave['history'].append(positions.copy())

                k1 = self._sample_velocity(data, positions)
                mid = positions + 0.5 * dt * k1
                mid[:, 0] = np.clip(mid[:, 0], x_min, x_max)
                mid[:, 1] = np.clip(mid[:, 1], y_min, y_max)
                k2 = self._sample_velocity(data, mid)
                spds = np.linalg.norm(k2, axis=1).astype(np.float32)
                wave['speeds'].append(spds)

                next_positions = positions + dt * k2
                outside = (
                    (next_positions[:, 0] < x_min)
                    | (next_positions[:, 0] > x_max)
                    | (next_positions[:, 1] < y_min)
                    | (next_positions[:, 1] > y_max)
                )
                inside = self._points_inside_obstacles(next_positions)
                local_t = frame_idx - wave['spawn_frame']
                stalled = spds < self.config.particle_respawn_speed_threshold
                respawn_mask = outside | inside | (stalled & (local_t > 2))

                next_positions[:, 0] = np.clip(next_positions[:, 0], x_min, x_max)
                next_positions[:, 1] = np.clip(next_positions[:, 1], y_min, y_max)

                if np.any(respawn_mask):
                    n_inlet = max(1, self._particle_inlet_seed_count)
                    seed_idx = rng.integers(0, n_inlet, size=int(respawn_mask.sum()))
                    next_positions[respawn_mask] = self._particle_seed_pool[seed_idx]
                    wave['pending_respawn_mask'][respawn_mask] = True

                wave['positions'] = next_positions.astype(np.float32, copy=False)

        for wave in active_waves:
            if not wave['history']:
                continue
            self._particle_waves.append({
                'spawn_frame': wave['spawn_frame'],
                'history': np.array(wave['history'], dtype=np.float32),
                'speeds': np.array(wave['speeds'], dtype=np.float32),
                'respawns': np.array(wave['respawns'], dtype=bool),
            })

        if self._particle_waves:
            w0 = self._particle_waves[0]
            self._particle_positions = w0['history']
            self._particle_speeds = w0['speeds']
            self._particle_respawns = w0['respawns']
        else:
            self._particle_positions = np.zeros((0, 0, 2), dtype=np.float32)
            self._particle_speeds = np.zeros((0, 0), dtype=np.float32)
            self._particle_respawns = np.zeros((0, 0), dtype=bool)

    def _update_particle_visuals(self, frame_idx: int):
        if self._particle_poly is None or self._particle_trail_poly is None:
            return

        if self._is_live:
            if self._live_particle_waves:
                points_xy = np.concatenate([w['positions'] for w in self._live_particle_waves], axis=0)
                point_speeds = np.concatenate([w['speeds'] for w in self._live_particle_waves], axis=0)
            else:
                points_xy = self._particle_seed_pool
                point_speeds = np.zeros(points_xy.shape[0], dtype=np.float32)
        else:
            if self._particle_waves:
                frame_idx = int(np.clip(frame_idx, 0, self.num_frames - 1))
                xy_parts, spd_parts = [], []
                for wave in self._particle_waves:
                    local_t = frame_idx - wave['spawn_frame']
                    if 0 <= local_t < wave['history'].shape[0]:
                        xy_parts.append(wave['history'][local_t])
                        spd_parts.append(wave['speeds'][local_t])
                if xy_parts:
                    points_xy = np.concatenate(xy_parts, axis=0)
                    point_speeds = np.concatenate(spd_parts, axis=0)
                else:
                    points_xy = np.empty((0, 2), dtype=np.float32)
                    point_speeds = np.empty(0, dtype=np.float32)
            else:
                points_xy = np.empty((0, 2), dtype=np.float32)
                point_speeds = np.empty(0, dtype=np.float32)

        wall_buf = self.config.wall_buffer_cells * self.config.dy
        y_lo = wall_buf
        y_hi = self.config.domain_height - wall_buf

        particle_points = vtk.vtkPoints()
        particle_verts = vtk.vtkCellArray()
        particle_speeds = vtk.vtkFloatArray()
        particle_speeds.SetName("particle_speed")

        for xy, speed in zip(points_xy, point_speeds):
            if self._obstacles and self._point_in_obstacle(float(xy[0]), float(xy[1])):
                continue
            if float(xy[1]) < y_lo or float(xy[1]) > y_hi:
                continue
            pid = particle_points.InsertNextPoint(float(xy[0]), float(xy[1]), 0.0)
            particle_verts.InsertNextCell(1)
            particle_verts.InsertCellPoint(pid)
            particle_speeds.InsertNextValue(float(speed))

        self._particle_poly.SetPoints(particle_points)
        self._particle_poly.SetVerts(particle_verts)
        pd = self._particle_poly.GetPointData()
        existing = pd.GetArray("particle_speed")
        if existing is None:
            pd.AddArray(particle_speeds)
        else:
            existing.DeepCopy(particle_speeds)
            existing.Modified()
        pd.SetActiveScalars("particle_speed")
        self._particle_poly.Modified()

        trail_points = vtk.vtkPoints()
        trail_lines = vtk.vtkCellArray()
        trail_speeds = vtk.vtkFloatArray()
        trail_speeds.SetName("particle_speed")

        if not self._is_live and self._particle_waves:
            jump_thresh = self.config.particle_respawn_jump_threshold
            for wave in self._particle_waves:
                local_frame = frame_idx - wave['spawn_frame']
                if local_frame < 0:
                    continue
                local_frame = min(local_frame, wave['history'].shape[0] - 1)
                start_local = max(0, local_frame - self.config.particle_trail_length + 1)
                wave_history = wave['history']
                wave_speeds_arr = wave['speeds']
                wave_respawns = wave['respawns']

                for particle_idx in range(wave_history.shape[1]):
                    current_segment = []
                    for lt in range(start_local, local_frame + 1):
                        if wave_respawns[lt, particle_idx] and current_segment:
                            self._append_trail_segment(current_segment, trail_points, trail_lines, trail_speeds)
                            current_segment = []

                        xy = wave_history[lt, particle_idx]
                        current_segment.append((float(xy[0]), float(xy[1]), 0.0, float(wave_speeds_arr[lt, particle_idx])))

                        if lt < local_frame:
                            nxt = wave_history[lt + 1, particle_idx]
                            jump = np.linalg.norm(nxt - xy)
                            if jump > jump_thresh:
                                self._append_trail_segment(current_segment, trail_points, trail_lines, trail_speeds)
                                current_segment = []
                    self._append_trail_segment(current_segment, trail_points, trail_lines, trail_speeds)

        self._particle_trail_poly.SetPoints(trail_points)
        self._particle_trail_poly.SetLines(trail_lines)
        trail_pd = self._particle_trail_poly.GetPointData()
        existing_trail = trail_pd.GetArray("particle_speed")
        if existing_trail is None:
            trail_pd.AddArray(trail_speeds)
        else:
            existing_trail.DeepCopy(trail_speeds)
            existing_trail.Modified()
        trail_pd.SetActiveScalars("particle_speed")
        self._particle_trail_poly.Modified()

    def _append_trail_segment(self, segment, points, lines, scalars):
        if len(segment) < 2:
            return
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(segment))
        for local_idx, (x, y, z, speed) in enumerate(segment):
            pid = points.InsertNextPoint(x, y, z)
            polyline.GetPointIds().SetId(local_idx, pid)
            scalars.InsertNextValue(speed)
        lines.InsertNextCell(polyline)