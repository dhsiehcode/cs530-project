import os
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from config import SimConfig, PlacedObstacle
from simulation.obstacles import create_obstacle_actor


class VTKPipeline:
    """Manages the full VTK rendering pipeline."""

    SCALAR_FIELDS = ("h", "speed", "vorticity")
    SCALAR_LABELS = {
        "h": "Height (m)",
        "speed": "Speed (m/s)",
        "vorticity": "Vorticity (1/s)",
    }

    def __init__(self, config: SimConfig, renderer: vtk.vtkRenderer):
        self.config = config
        self.renderer = renderer
        self.data_dir = ""
        self.num_frames = 0
        self.active_field = "h"

        self.show_surface = True
        self.show_glyphs = True
        self.show_contours = True
        self.show_streamlines = True

        self.reader = vtk.vtkXMLImageDataReader()
        self._live_image = None
        self._live_producer = None
        self._live_arrays = {}
        self._live_buffers = {}
        self._live_frame_counter = 0
        self._source = self.reader
        self._is_live = False

        self.surface_actor = None
        self.glyph_actor = None
        self.contour_actor = None
        self.streamline_actor = None
        self.obstacle_actors: list = []
        self.scalar_bars = {}
        self.corner_annotation = None
        self._coordinate_picker = vtk.vtkWorldPointPicker()
        self._coordinate_interactor = None
        self._mouse_move_observer = None
        self._last_coordinate_text = "X: --  Y: --"

        self._animating = False
        self._obstacles: list[PlacedObstacle] = []

        self._build_color_maps()
        self.scalar_ranges = {
            "h": (0.0, 1.0),
            "speed": (0.0, 1.5),
            "vorticity": (-10.0, 10.0),
        }

    def _get_source_port(self):
        return self._source.GetOutputPort()

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
                    angle = np.float32(np.deg2rad(defn.angle))
                    axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                    half_len = 0.5 * float(defn.length)
                    a = center - half_len * axis
                    rel = points - a
                    t = np.clip(rel @ axis, 0.0, float(defn.length))
                    closest = a + np.outer(t, axis)
                    core_radius = max(defn.radius * 1.05, 0.5 * min(self.config.dx, self.config.dy))
                    shell_radius = defn.radius + shell_pad

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
        viz_velocity = np.column_stack(
            [vec[:, 0], vec[:, 1], np.zeros(vec.shape[0], dtype=np.float32)]
        )

        self._upsert_array(pd, "viz_vx", vec[:, 0])
        self._upsert_array(pd, "viz_vy", vec[:, 1])
        self._upsert_array(pd, "viz_speed", viz_speed)
        self._upsert_array(pd, "viz_velocity", viz_velocity, components=3)

        pd.SetActiveScalars("h")
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
        self._build_pipeline()
        self._add_obstacles(obstacles)
        self._setup_camera()

    def start_live_mode(self, nx: int, ny: int, dx: float, dy: float, obstacles: list):
        self._is_live = True
        self._live_frame_counter = 0
        self._obstacles = list(obstacles)

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
            a = numpy_to_vtk(buffer, deep=True)
            a.SetName(name)
            pd.AddArray(a)
            self._live_arrays[name] = a
            self._live_buffers[name] = vtk_to_numpy(a)

        vec = numpy_to_vtk(np.zeros((n, 3), dtype=np.float32), deep=True)
        vec.SetName("velocity")
        pd.AddArray(vec)
        self._live_arrays["velocity"] = vec
        self._live_buffers["velocity"] = vtk_to_numpy(vec)
        pd.SetActiveScalars("h")
        pd.SetActiveVectors("velocity")

        self._live_producer = vtk.vtkTrivialProducer()
        self._live_producer.SetOutput(self._live_image)
        self._source = self._live_producer

        self.scalar_ranges = {
            "h": (0.0, 1.0),
            "speed": (0.0, 2.0),
            "vorticity": (-10.0, 10.0),
        }

        self._build_pipeline()
        self._add_obstacles(obstacles)
        self._setup_camera()

    def update_live_frame(self, frame_data: dict, render: bool = True):
        if self._live_image is None:
            return

        for name in ("h", "eta", "vx", "vy", "speed", "vorticity", "pressure"):
            np.copyto(
                self._live_buffers[name],
                frame_data[name].flatten(order="F").astype(np.float32, copy=False),
            )
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
                a = pd.GetArray(name)
                if a:
                    lo_d, hi_d = a.GetRange()
                    lo_c, hi_c = self.scalar_ranges[name]
                    if name == "vorticity":
                        mx = max(abs(lo_d), abs(hi_d), abs(lo_c), abs(hi_c), 0.1)
                        self.scalar_ranges[name] = (-mx, mx)
                    else:
                        self.scalar_ranges[name] = (min(lo_c, lo_d), max(hi_c, hi_d))

            if hasattr(self, "_surface_mapper") and self._surface_mapper:
                lo, hi = self.scalar_ranges[self.active_field]
                self._surface_mapper.SetScalarRange(lo, hi)
            self._sync_ranges()

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
        self.renderer.GetRenderWindow().Render()

    def set_scalar_field(self, field_name: str):
        if field_name not in self.SCALAR_FIELDS:
            return
        self.active_field = field_name
        if not hasattr(self, "_surface_mapper"):
            return
        lo, hi = self.scalar_ranges[field_name]
        self._surface_mapper.SelectColorArray(field_name)
        self._surface_mapper.SetScalarRange(lo, hi)
        self._surface_mapper.SetLookupTable(self.ctfs[field_name])
        if self.scalar_bars.get("surface"):
            self.scalar_bars["surface"].SetLookupTable(self.ctfs[field_name])
            self.scalar_bars["surface"].SetTitle(self.SCALAR_LABELS[field_name])
        self.renderer.GetRenderWindow().Render()

    def set_layer_visibility(self, layer: str, visible: bool):
        actors = {
            "surface": self.surface_actor,
            "glyphs": self.glyph_actor,
            "contours": self.contour_actor,
            "streamlines": self.streamline_actor,
        }
        actor = actors.get(layer)
        if actor:
            actor.SetVisibility(visible)
            if layer == "surface":
                self.show_surface = visible
            elif layer == "glyphs":
                self.show_glyphs = visible
            elif layer == "contours":
                self.show_contours = visible
            elif layer == "streamlines":
                self.show_streamlines = visible
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

        def _on_mouse_move(obj, event):
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
        self.glyph_actor = None
        self.contour_actor = None
        self.streamline_actor = None
        self._riverbed_actor = None
        self._surface_mapper = None
        self._glyph_mapper = None
        self._contour_mapper = None
        self._streamline_mapper = None
        self.obstacle_actors.clear()
        self.scalar_bars = {}
        self.corner_annotation = None

    def update_obstacles(self, obstacles: list):
        self._obstacles = list(obstacles)
        for actor in self.obstacle_actors:
            self.renderer.RemoveActor(actor)
        self.obstacle_actors.clear()
        self._add_obstacles(obstacles)
        self._apply_obstacle_aware_flow()
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
        if getattr(self, "_glyph_mapper", None):
            self._glyph_mapper.SetScalarRange(lo, hi)
        if getattr(self, "_streamline_mapper", None):
            self._streamline_mapper.SetScalarRange(lo, hi)
        if getattr(self, "_contour_mapper", None):
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
        ctf_s.AddRGBPoint(0.0, 0.05, 0.10, 0.50)
        ctf_s.AddRGBPoint(0.20, 0.08, 0.40, 0.70)
        ctf_s.AddRGBPoint(0.40, 0.10, 0.65, 0.60)
        ctf_s.AddRGBPoint(0.60, 0.40, 0.80, 0.20)
        ctf_s.AddRGBPoint(0.80, 0.90, 0.75, 0.10)
        ctf_s.AddRGBPoint(1.0, 0.85, 0.15, 0.08)
        self.ctfs["speed"] = ctf_s

        ctf_v = vtk.vtkColorTransferFunction()
        ctf_v.SetColorSpaceToDiverging()
        ctf_v.AddRGBPoint(-10.0, 0.231, 0.298, 0.753)
        ctf_v.AddRGBPoint(0.0, 0.865, 0.865, 0.865)
        ctf_v.AddRGBPoint(10.0, 0.706, 0.016, 0.150)
        self.ctfs["vorticity"] = ctf_v

    def _build_pipeline(self):
        self.clear()
        self._build_riverbed()
        self._build_surface()
        self._build_contours()
        self._build_glyphs()
        self._build_streamlines()
        self._build_scalar_bars()
        self._setup_lighting()

    def _build_riverbed(self):
        w = self.config.domain_width
        h = self.config.domain_height
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(0, 0, -0.01)
        plane.SetPoint1(w, 0, -0.01)
        plane.SetPoint2(0, h, -0.01)
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

    def _surface_offset_filter(self, input_port):
        geom = vtk.vtkImageDataGeometryFilter()
        geom.SetInputConnection(input_port)

        calc = vtk.vtkArrayCalculator()
        calc.SetInputConnection(geom.GetOutputPort())
        calc.SetAttributeTypeToPointData()
        calc.AddScalarArrayName("eta", 0)
        calc.SetResultArrayName("surface_offset")
        calc.SetFunction(f"eta - {self.config.h0}")

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
        warp.SetScaleFactor(self.config.warp_scale)

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
        actor.AddPosition(0, 0, 0.02)
        prop = actor.GetProperty()
        prop.SetOpacity(0.90)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(40)
        prop.SetSpecularColor(1.0, 1.0, 1.0)
        prop.SetDiffuse(0.7)
        prop.SetAmbient(0.16)
        actor.SetVisibility(self.show_surface)

        self.renderer.AddActor(actor)
        self.surface_actor = actor
        self._surface_mapper = mapper

    def _build_glyphs(self):
        sub = vtk.vtkExtractVOI()
        sub.SetInputConnection(self._get_source_port())
        sample_rate = 40 if self._is_live else 14
        sub.SetSampleRate(sample_rate, sample_rate, 1)

        assign = self._surface_offset_filter(sub.GetOutputPort())

        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(assign.GetOutputPort())
        warp.SetScaleFactor(self.config.warp_scale)

        assign_v = vtk.vtkAssignAttribute()
        assign_v.SetInputConnection(warp.GetOutputPort())
        assign_v.Assign(
            "viz_velocity",
            vtk.vtkDataSetAttributes.VECTORS,
            vtk.vtkAssignAttribute.POINT_DATA,
        )

        arrow = vtk.vtkArrowSource()
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.2)
        arrow.SetShaftRadius(0.08)

        glyph_threshold = vtk.vtkThreshold()
        glyph_threshold.SetInputConnection(assign_v.GetOutputPort())
        glyph_threshold.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "viz_speed"
        )
        glyph_threshold.SetLowerThreshold(0.02)

        glyph_geom = vtk.vtkGeometryFilter()
        glyph_geom.SetInputConnection(glyph_threshold.GetOutputPort())

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputConnection(glyph_geom.GetOutputPort())
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(0.75 if self._is_live else 0.45)
        glyph.OrientOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("viz_speed")
        lo, hi = self.scalar_ranges["speed"]
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["speed"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(0, 0, 0.12)
        actor.GetProperty().SetOpacity(1.0)
        actor.SetVisibility(self.show_glyphs)

        self.renderer.AddActor(actor)
        self.glyph_actor = actor
        self._glyph_mapper = mapper

    def _build_contours(self):
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(self._get_source_port())
        contour.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vorticity"
        )
        lo, hi = self.scalar_ranges["vorticity"]
        contour.GenerateValues(8 if self._is_live else 12, lo, hi)

        transform = vtk.vtkTransform()
        transform.Translate(0, 0, 0.07)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(contour.GetOutputPort())
        tf.SetTransform(transform)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.03 if self._is_live else 0.018)
        tube.SetNumberOfSides(8)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["vorticity"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.95)
        actor.SetVisibility(self.show_contours)

        self.renderer.AddActor(actor)
        self.contour_actor = actor
        self._contour_mapper = mapper

    def _build_streamlines(self):
        assign_v = vtk.vtkAssignAttribute()
        assign_v.SetInputConnection(self._get_source_port())
        assign_v.Assign(
            "viz_velocity",
            vtk.vtkDataSetAttributes.VECTORS,
            vtk.vtkAssignAttribute.POINT_DATA,
        )

        y0 = self.config.dy
        y1 = self.config.domain_height - self.config.dy

        # Seed across the full inlet height, slightly inside the domain.
        seeds = vtk.vtkLineSource()
        seeds.SetPoint1(self.config.dx * 1.5, y0, 0.0)
        seeds.SetPoint2(self.config.dx * 1.5, y1, 0.0)
        seeds.SetResolution(36 if self._is_live else 56)

        tracer = vtk.vtkStreamTracer()
        tracer.SetInputConnection(assign_v.GetOutputPort())
        tracer.SetSourceConnection(seeds.GetOutputPort())
        tracer.SetMaximumPropagation(self.config.domain_width * 3.0)
        tracer.SetIntegrationDirectionToForward()
        tracer.SetIntegratorTypeToRungeKutta4()
        tracer.SetInitialIntegrationStep(self.config.dx * 0.4)
        tracer.SetMinimumIntegrationStep(self.config.dx * 0.05)
        tracer.SetMaximumIntegrationStep(self.config.dx * 1.2)
        tracer.SetMaximumNumberOfSteps(4000 if self._is_live else 9000)
        tracer.SetTerminalSpeed(1.0e-6)
        tracer.SetComputeVorticity(False)

        transform = vtk.vtkTransform()
        transform.Translate(0, 0, 0.17)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(tracer.GetOutputPort())
        tf.SetTransform(transform)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.032 if self._is_live else 0.022)
        tube.SetNumberOfSides(8)
        tube.CappingOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("IntegrationTime")
        mapper.SetScalarRange(0.0, self.config.domain_width * 3.0)
        mapper.SetLookupTable(self.ctfs["speed"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(1.0)
        actor.SetVisibility(self.show_streamlines)

        self.renderer.AddActor(actor)
        self.streamline_actor = actor
        self._streamline_mapper = mapper

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
        glyph_bar = self._make_scalar_bar("Glyph Speed", self.ctfs["speed"], 0.01, 0.67, 0.24)
        streamline_bar = self._make_scalar_bar("Streamline Speed", self.ctfs["speed"], 0.01, 0.38, 0.24)
        contour_bar = self._make_scalar_bar("Vorticity", self.ctfs["vorticity"], 0.01, 0.09, 0.24)

        for bar in (surface_bar, glyph_bar, streamline_bar, contour_bar):
            self.renderer.AddActor2D(bar)

        self.scalar_bars = {
            "surface": surface_bar,
            "glyphs": glyph_bar,
            "streamlines": streamline_bar,
            "contours": contour_bar,
        }
        self._update_scalar_bar_visibility()

    def _update_scalar_bar_visibility(self):
        if not self.scalar_bars:
            return
        self.scalar_bars["surface"].SetVisibility(self.show_surface)
        self.scalar_bars["glyphs"].SetVisibility(self.show_glyphs)
        self.scalar_bars["streamlines"].SetVisibility(self.show_streamlines)
        self.scalar_bars["contours"].SetVisibility(self.show_contours)

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