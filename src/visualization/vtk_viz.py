import os
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from config import SimConfig, PlacedObstacle
from simulation.obstacles import create_obstacle_actor


class VTKPipeline:
    """Manages the full VTK rendering pipeline."""

    SCALAR_FIELDS = ("h", "speed", "vorticity")
    SCALAR_LABELS = {"h": "Height (m)", "speed": "Speed (m/s)",
                     "vorticity": "Vorticity (1/s)"}

    def __init__(self, config: SimConfig, renderer: vtk.vtkRenderer):
        self.config = config
        self.renderer = renderer
        self.data_dir = ""
        self.num_frames = 0
        self.active_field = "h"

        # visibility flags
        self.show_surface = True
        self.show_glyphs = True
        self.show_contours = True
        self.show_streamlines = True

        # VTK objects (populated by setup_pipeline)
        self.reader = vtk.vtkXMLImageDataReader()
        self._live_image = None
        self._live_producer = None
        self._live_arrays = {}
        self._live_buffers = {}
        self._live_frame_counter = 0
        self._source = self.reader          # current pipeline source
        self._is_live = False

        self.surface_actor = None
        self.glyph_actor = None
        self.contour_actor = None
        self.streamline_actor = None
        self.obstacle_actors: list = []
        self.scalar_bar = None
        self.corner_annotation = None
        self._coordinate_picker = vtk.vtkWorldPointPicker()
        self._coordinate_interactor = None
        self._mouse_move_observer = None
        self._last_coordinate_text = "X: --  Y: --"

        # Animation state — when True, expensive filters are skipped
        self._animating = False
        # Whether streamlines/contours were visible before animation started
        self._pre_anim_streamlines = True
        self._pre_anim_contours = True

        # color maps
        self._build_color_maps()
        # estimated scalar ranges – updated after first frame load
        self.scalar_ranges = {"h": (0.0, 1.0), "speed": (0.0, 1.5),
                              "vorticity": (-10.0, 10.0)}

    # ------------------------------------------------------------------ #
    #  Source abstraction                                                  #
    # ------------------------------------------------------------------ #
    def _get_source_port(self):
        """Return the output port of the current data source."""
        return self._source.GetOutputPort()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def load_simulation(self, data_dir: str, num_frames: int,
                        obstacles: list):
        """Load a completed simulation and build the pipeline (file mode)."""
        self._is_live = False
        self._source = self.reader
        self.data_dir = data_dir
        self.num_frames = num_frames

        self._load_frame(0)
        self._estimate_ranges()
        self._build_pipeline()
        self._add_obstacles(obstacles)
        self._setup_camera()

    # ---- live preview ------------------------------------------------ #
    def start_live_mode(self, nx: int, ny: int, dx: float, dy: float,
                        obstacles: list):
        """Switch pipeline to live data mode at the given resolution."""
        self._is_live = True
        self._live_frame_counter = 0

        # Create an in-memory vtkImageData
        self._live_image = vtk.vtkImageData()
        self._live_image.SetDimensions(nx, ny, 1)
        self._live_image.SetSpacing(dx, dy, 1.0)
        self._live_image.SetOrigin(0, 0, 0)

        # Seed with zeros so the pipeline has valid geometry
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

        # Reasonable default ranges for live mode
        self.scalar_ranges = {"h": (0.0, 1.0), "speed": (0.0, 2.0),
                              "vorticity": (-10.0, 10.0)}

        self._build_pipeline()
        self._add_obstacles(obstacles)
        self._setup_camera()

    def update_live_frame(self, frame_data: dict, render: bool = True):
        """Push new solver output into the live image and optionally re-render."""
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

        pd = self._live_image.GetPointData()
        pd.SetActiveScalars("h")
        pd.SetActiveVectors("velocity")
        self._live_image.Modified()

        self._live_frame_counter += 1
        if (self._live_frame_counter %
                max(1, self.config.live_preview_range_update_interval) == 0):
            # Adaptive range update — expand ranges to fit data
            for name in self.SCALAR_FIELDS:
                a = pd.GetArray(name)
                if a:
                    lo_d, hi_d = a.GetRange()
                    lo_c, hi_c = self.scalar_ranges[name]
                    if name == "vorticity":
                        mx = max(abs(lo_d), abs(hi_d), abs(lo_c), abs(hi_c), 0.1)
                        self.scalar_ranges[name] = (-mx, mx)
                    else:
                        self.scalar_ranges[name] = (min(lo_c, lo_d),
                                                    max(hi_c, hi_d))

            # apply updated range to surface mapper
            if hasattr(self, "_surface_mapper"):
                lo, hi = self.scalar_ranges[self.active_field]
                self._surface_mapper.SetScalarRange(lo, hi)
            # keep glyph/streamline speed ranges in sync
            self._sync_speed_ranges()

        if render:
            self.renderer.GetRenderWindow().Render()

    def stop_live_mode(self):
        """Tear down the live pipeline."""
        self._is_live = False
        self._live_image = None
        self._live_producer = None
        self._live_arrays = {}
        self._live_buffers = {}
        self._live_frame_counter = 0
        self._source = self.reader
        self.clear()

    # ---- shared API -------------------------------------------------- #
    def set_animating(self, playing: bool):
        """Toggle animation mode — hides expensive layers during playback.

        vtkStreamTracer and vtkContourFilter re-execute on every pipeline
        update.  Hiding them during playback prevents the GPU/CPU from
        recomputing streamlines for every single frame, which is the main
        cause of playback freezes.
        """
        if playing == self._animating:
            return
        self._animating = playing
        if playing:
            # Save current visibility and hide expensive layers
            if self.streamline_actor:
                self._pre_anim_streamlines = self.streamline_actor.GetVisibility()
                self.streamline_actor.SetVisibility(False)
            if self.contour_actor:
                self._pre_anim_contours = self.contour_actor.GetVisibility()
                self.contour_actor.SetVisibility(False)
        else:
            # Restore visibility when paused
            if self.streamline_actor:
                self.streamline_actor.SetVisibility(self._pre_anim_streamlines)
            if self.contour_actor:
                self.contour_actor.SetVisibility(self._pre_anim_contours)
            # Force a full pipeline update now that we're paused
            self.renderer.GetRenderWindow().Render()

    def set_frame(self, idx: int):
        """Switch the reader to a different time frame and re-render."""
        if not self._load_frame(idx):
            return
        self._refresh_active_arrays()
        self.renderer.GetRenderWindow().Render()

    def set_scalar_field(self, field_name: str):
        """Change the colour-mapped scalar field on the surface."""
        if field_name not in self.SCALAR_FIELDS:
            return
        self.active_field = field_name
        if not hasattr(self, "_surface_mapper"):
            return
        lo, hi = self.scalar_ranges[field_name]
        self._surface_mapper.SelectColorArray(field_name)
        self._surface_mapper.SetScalarRange(lo, hi)
        self._surface_mapper.SetLookupTable(self.ctfs[field_name])
        if self.scalar_bar:
            self.scalar_bar.SetLookupTable(self.ctfs[field_name])
            self.scalar_bar.SetTitle(self.SCALAR_LABELS[field_name])
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
            self.renderer.GetRenderWindow().Render()

    def setup_coordinate_display(self, interactor):
        """Add a corner annotation that tracks mouse world coordinates."""
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
        self._mouse_move_observer = interactor.AddObserver(
            "MouseMoveEvent", _on_mouse_move)

    def clear(self):
        """Remove all actors from the renderer."""
        self.renderer.RemoveAllViewProps()
        self.surface_actor = None
        self.glyph_actor = None
        self.contour_actor = None
        self.streamline_actor = None
        self._riverbed_actor = None
        self._surface_mapper = None
        self._glyph_mapper = None
        self._streamline_mapper = None
        self.obstacle_actors.clear()
        self.scalar_bar = None
        self.corner_annotation = None

    def update_obstacles(self, obstacles: list):
        """Replace obstacle actors with the given obstacle list."""
        for actor in self.obstacle_actors:
            self.renderer.RemoveActor(actor)
        self.obstacle_actors.clear()
        self._add_obstacles(obstacles)
        self.renderer.GetRenderWindow().Render()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _load_frame(self, idx: int) -> bool:
        path = os.path.join(self.data_dir, f"frame_{idx:04d}.vti")
        if not os.path.exists(path):
            return False
        self.reader.SetFileName(path)
        self.reader.Update()
        self._refresh_active_arrays()
        return True

    def _refresh_active_arrays(self):
        data = self.reader.GetOutput()
        if data:
            data.GetPointData().SetActiveScalars("h")
            data.GetPointData().SetActiveVectors("velocity")

    def _estimate_ranges(self):
        """Scan the first frame to set reasonable colour-map ranges."""
        if self._is_live:
            data = self._live_image
        else:
            data = self.reader.GetOutput()
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
        # Propagate speed range to glyph/streamline mappers
        self._sync_speed_ranges()

    def _sync_speed_ranges(self):
        """Push the current speed range to glyph and streamline mappers."""
        lo, hi = self.scalar_ranges["speed"]
        if hasattr(self, "_glyph_mapper") and self._glyph_mapper:
            self._glyph_mapper.SetScalarRange(lo, hi)
        if hasattr(self, "_streamline_mapper") and self._streamline_mapper:
            self._streamline_mapper.SetScalarRange(lo, hi)

    # ---- colour maps ------------------------------------------------- #
    def _build_color_maps(self):
        self.ctfs = {}

        # height – deep blue (deep) → turquoise → light cyan (shallow peak)
        ctf_h = vtk.vtkColorTransferFunction()
        ctf_h.SetColorSpaceToLab()
        ctf_h.AddRGBPoint(0.0,  0.02, 0.08, 0.30)   # dark navy
        ctf_h.AddRGBPoint(0.25, 0.04, 0.18, 0.52)   # ocean blue
        ctf_h.AddRGBPoint(0.50, 0.06, 0.38, 0.66)   # medium blue
        ctf_h.AddRGBPoint(0.70, 0.10, 0.58, 0.74)   # teal
        ctf_h.AddRGBPoint(0.85, 0.25, 0.72, 0.82)   # turquoise
        ctf_h.AddRGBPoint(1.0,  0.55, 0.88, 0.94)   # light cyan
        self.ctfs["h"] = ctf_h

        # speed – dark blue (still) → cyan → green → yellow → red (fast)
        ctf_s = vtk.vtkColorTransferFunction()
        ctf_s.SetColorSpaceToLab()
        ctf_s.AddRGBPoint(0.0,  0.05, 0.10, 0.50)   # dark blue (still)
        ctf_s.AddRGBPoint(0.20, 0.08, 0.40, 0.70)   # blue
        ctf_s.AddRGBPoint(0.40, 0.10, 0.65, 0.60)   # teal-green
        ctf_s.AddRGBPoint(0.60, 0.40, 0.80, 0.20)   # green-yellow
        ctf_s.AddRGBPoint(0.80, 0.90, 0.75, 0.10)   # yellow
        ctf_s.AddRGBPoint(1.0,  0.85, 0.15, 0.08)   # red (fast)
        self.ctfs["speed"] = ctf_s

        # vorticity – diverging blue-white-red
        ctf_v = vtk.vtkColorTransferFunction()
        ctf_v.SetColorSpaceToDiverging()
        ctf_v.AddRGBPoint(-10.0, 0.231, 0.298, 0.753)
        ctf_v.AddRGBPoint(0.0,   0.865, 0.865, 0.865)
        ctf_v.AddRGBPoint(10.0,  0.706, 0.016, 0.150)
        self.ctfs["vorticity"] = ctf_v

    # ---- pipeline construction --------------------------------------- #
    def _build_pipeline(self):
        self.clear()
        self._build_riverbed()
        self._build_surface()
        self._build_glyphs()
        self._build_contours()
        self._build_streamlines()
        self._build_scalar_bar()
        self._setup_lighting()

    def _build_riverbed(self):
        """Flat textured plane at z=0 to give spatial context."""
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
        actor.GetProperty().SetColor(0.58, 0.50, 0.36)   # sandy brown
        actor.GetProperty().SetAmbient(0.4)
        actor.GetProperty().SetDiffuse(0.6)

        self.renderer.AddActor(actor)
        self._riverbed_actor = actor

    def _build_surface(self):
        geom = vtk.vtkImageDataGeometryFilter()
        geom.SetInputConnection(self._get_source_port())

        # Warp by η = h + b (free-surface elevation) so the surface is
        # physically flat at rest and only real perturbations appear as bumps.
        assign_eta = vtk.vtkAssignAttribute()
        assign_eta.SetInputConnection(geom.GetOutputPort())
        assign_eta.Assign("eta", vtk.vtkDataSetAttributes.SCALARS,
                          vtk.vtkAssignAttribute.POINT_DATA)

        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(assign_eta.GetOutputPort())
        warp.SetScaleFactor(self.config.warp_scale)

        # Normals for proper specular highlights (water sheen)
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
        prop = actor.GetProperty()
        prop.SetOpacity(0.88)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(40)
        prop.SetSpecularColor(1.0, 1.0, 1.0)
        prop.SetDiffuse(0.7)
        prop.SetAmbient(0.15)
        actor.SetVisibility(self.show_surface)

        self.renderer.AddActor(actor)
        self.surface_actor = actor
        self._surface_mapper = mapper
        self._surface_warp = warp

    def _build_glyphs(self):
        sub = vtk.vtkExtractVOI()
        sub.SetInputConnection(self._get_source_port())
        sample_rate = 48 if self._is_live else 16
        sub.SetSampleRate(sample_rate, sample_rate, 1)

        geom = vtk.vtkImageDataGeometryFilter()
        geom.SetInputConnection(sub.GetOutputPort())

        assign_eta = vtk.vtkAssignAttribute()
        assign_eta.SetInputConnection(geom.GetOutputPort())
        assign_eta.Assign("eta", vtk.vtkDataSetAttributes.SCALARS,
                          vtk.vtkAssignAttribute.POINT_DATA)

        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(assign_eta.GetOutputPort())
        warp.SetScaleFactor(self.config.warp_scale)

        assign_v = vtk.vtkAssignAttribute()
        assign_v.SetInputConnection(warp.GetOutputPort())
        assign_v.Assign("velocity", vtk.vtkDataSetAttributes.VECTORS,
                        vtk.vtkAssignAttribute.POINT_DATA)

        arrow = vtk.vtkArrowSource()
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.2)
        arrow.SetShaftRadius(0.08)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputConnection(assign_v.GetOutputPort())
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(0.7 if self._is_live else 0.4)
        glyph.OrientOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        # Color arrows by speed so they show flow intensity
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("speed")
        lo, hi = self.scalar_ranges["speed"]
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["speed"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(0, 0, self.config.glyph_z_offset)
        actor.GetProperty().SetOpacity(0.9)
        actor.SetVisibility(self.show_glyphs)

        self.renderer.AddActor(actor)
        self.glyph_actor = actor
        self._glyph_mapper = mapper

    def _build_contours(self):
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(self._get_source_port())
        contour.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vorticity")
        lo, hi = self.scalar_ranges["vorticity"]
        contour.GenerateValues(8 if self._is_live else 12, lo, hi)

        z_offset = self.config.h0 * self.config.warp_scale + 0.15
        transform = vtk.vtkTransform()
        transform.Translate(0, 0, z_offset)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(contour.GetOutputPort())
        tf.SetTransform(transform)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.035 if self._is_live else 0.02)
        tube.SetNumberOfSides(6)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["vorticity"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(0, 0, self.config.contour_z_offset)
        actor.GetProperty().SetOpacity(0.85 if self._is_live else 0.75)
        actor.SetVisibility(self.show_contours)

        self.renderer.AddActor(actor)
        self.contour_actor = actor

    def _build_streamlines(self):
        assign_v = vtk.vtkAssignAttribute()
        assign_v.SetInputConnection(self._get_source_port())
        assign_v.Assign("velocity", vtk.vtkDataSetAttributes.VECTORS,
                        vtk.vtkAssignAttribute.POINT_DATA)

        seeds = vtk.vtkLineSource()
        seeds.SetPoint1(self.config.dx * 2, self.config.dy * 5, 0)
        seeds.SetPoint2(self.config.dx * 2,
                        self.config.domain_height - self.config.dy * 5, 0)
        seeds.SetResolution(8 if self._is_live else 12)

        tracer = vtk.vtkStreamTracer()
        tracer.SetInputConnection(assign_v.GetOutputPort())
        tracer.SetSourceConnection(seeds.GetOutputPort())
        tracer.SetMaximumPropagation(
            self.config.domain_width if self._is_live else self.config.domain_width * 1.5)
        tracer.SetIntegrationDirectionToForward()
        tracer.SetIntegratorTypeToRungeKutta4()
        tracer.SetMaximumNumberOfSteps(800 if self._is_live else 2000)

        z_offset = self.config.h0 * self.config.warp_scale + 0.25
        transform = vtk.vtkTransform()
        transform.Translate(0, 0, z_offset)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(tracer.GetOutputPort())
        tf.SetTransform(transform)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.035 if self._is_live else 0.025)
        tube.SetNumberOfSides(6)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        # Color streamlines by speed to show flow acceleration
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("speed")
        lo, hi = self.scalar_ranges["speed"]
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["speed"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(0, 0, self.config.streamline_z_offset)
        actor.GetProperty().SetOpacity(0.85 if self._is_live else 0.75)
        actor.SetVisibility(self.show_streamlines)

        self.renderer.AddActor(actor)
        self.streamline_actor = actor
        self._streamline_mapper = mapper

    def _build_scalar_bar(self):
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(self.ctfs[self.active_field])
        bar.SetTitle(self.SCALAR_LABELS[self.active_field])
        bar.SetNumberOfLabels(5)
        bar.SetWidth(0.08)
        bar.SetHeight(0.4)
        bar.SetPosition(0.90, 0.05)
        bar.GetTitleTextProperty().SetFontSize(12)
        bar.GetTitleTextProperty().SetColor(1, 1, 1)
        bar.GetLabelTextProperty().SetColor(1, 1, 1)

        self.renderer.AddActor2D(bar)
        self.scalar_bar = bar

    # ---- obstacle actors --------------------------------------------- #
    def _add_obstacles(self, obstacles: list):
        for obs in obstacles:
            actor = create_obstacle_actor(obs, self.config.warp_scale)
            self.renderer.AddActor(actor)
            self.obstacle_actors.append(actor)

    # ---- lighting ---------------------------------------------------- #
    def _setup_lighting(self):
        """Add a key light and fill light for realistic water sheen."""
        self.renderer.RemoveAllLights()

        # Key light — high angle from upstream-right, warm white
        key = vtk.vtkLight()
        key.SetLightTypeToSceneLight()
        w = self.config.domain_width
        h = self.config.domain_height
        key.SetPosition(w * 0.3, -h * 0.5, h * 2.0)
        key.SetFocalPoint(w / 2, h / 2, 0)
        key.SetColor(1.0, 0.98, 0.94)
        key.SetIntensity(1.0)
        self.renderer.AddLight(key)

        # Fill light — soft from opposite side, cool tint
        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(w * 0.8, h * 1.5, h * 0.8)
        fill.SetFocalPoint(w / 2, h / 2, 0)
        fill.SetColor(0.85, 0.90, 1.0)
        fill.SetIntensity(0.4)
        self.renderer.AddLight(fill)

    # ---- camera ------------------------------------------------------ #
    def _setup_camera(self):
        w = self.config.domain_width
        h = self.config.domain_height
        cam = self.renderer.GetActiveCamera()
        # View from slightly upstream, elevated — like standing on a bridge
        cam.SetPosition(w * 0.25, -h * 0.7, h * 1.1)
        cam.SetFocalPoint(w * 0.55, h / 2, 0)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        # Gradient background: dark blue-grey at bottom, lighter at top
        self.renderer.SetBackground(0.12, 0.14, 0.22)
        self.renderer.SetBackground2(0.28, 0.35, 0.50)
        self.renderer.GradientBackgroundOn()
