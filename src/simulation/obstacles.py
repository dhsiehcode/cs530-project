"""
Obstacle utilities:
  - Build the bed-elevation array *b* for the Taichi solver.
  - Create VTK meshes for rendering obstacles in the scene.
"""
import math
import random
import numpy as np
import vtk

from config import SimConfig, PlacedObstacle


# ------------------------------------------------------------------ #
#  Shared helpers                                                     #
# ------------------------------------------------------------------ #
def _rock_seed(obs: PlacedObstacle) -> int:
    """Stable deterministic seed for one rock."""
    xi = int(round(obs.x * 1000.0))
    yi = int(round(obs.y * 1000.0))
    oid = int(getattr(obs, "obstacle_id", 0))
    return ((xi * 73856093) ^ (yi * 19349663) ^ (oid * 83492791)) & 0xFFFFFFFF


def _rock_params(obs: PlacedObstacle) -> dict:
    rng = random.Random(_rock_seed(obs))
    return {
        "phase1": rng.uniform(0.0, 2.0 * math.pi),
        "phase2": rng.uniform(0.0, 2.0 * math.pi),
        "phase3": rng.uniform(0.0, 2.0 * math.pi),
        "phase4": rng.uniform(0.0, 2.0 * math.pi),
        "phase5": rng.uniform(0.0, 2.0 * math.pi),
    }


def _rock_radius_scale(theta, params: dict):
    """Angularly varying radius scale for a mildly irregular 2-D rock footprint."""
    scale = (
        1.0
        + 0.08 * np.sin(3.0 * theta + params["phase1"])
        + 0.04 * np.sin(7.0 * theta + params["phase2"])
        + 0.02 * np.cos(11.0 * theta + params["phase3"])
    )
    return np.clip(scale, 0.88, 1.12)


# ------------------------------------------------------------------ #
#  Bed-elevation builder (numpy, vectorized)                         #
# ------------------------------------------------------------------ #
def build_bed_elevation(config: SimConfig, obstacles: list) -> np.ndarray:
    """Return the bed-elevation array b[nx, ny] with obstacle bumps/ridges."""
    b = np.zeros((config.nx, config.ny), dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(config.nx), np.arange(config.ny), indexing="ij")

    for obs in obstacles:
        defn = obs.definition
        cx = obs.x / config.dx
        cy = obs.y / config.dy

        if defn.kind == "rock":
            r_cells = defn.radius / config.dx
            dx_local = ii - cx
            dy_local = jj - cy
            dist = np.sqrt(dx_local ** 2 + dy_local ** 2)
            theta = np.arctan2(dy_local, dx_local)
            params = _rock_params(obs)

            jagged_r = r_cells * _rock_radius_scale(theta, params)
            norm = dist / np.maximum(jagged_r, 1.0e-6)
            mask = norm < 1.0

            radial_profile = 0.5 * (1.0 + np.cos(np.pi * np.minimum(norm, 1.0)))
            crest = 0.97 + 0.03 * np.cos(4.0 * theta + params["phase4"])
            vals = defn.height * radial_profile * crest
            b = np.maximum(b, vals * mask)

        elif defn.kind == "log":
            r_cells = defn.radius / config.dx
            length_cells = defn.length / config.dx
            angle_rad = math.radians(defn.angle + 90)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            dx_local = (ii - cx) * cos_a + (jj - cy) * sin_a
            dy_local = -(ii - cx) * sin_a + (jj - cy) * cos_a

            mask_along = np.abs(dx_local) < length_cells / 2
            mask_cross = np.abs(dy_local) < r_cells
            mask = mask_along & mask_cross
            cross_profile = 0.5 * (1.0 + np.cos(np.pi * np.minimum(
                np.abs(dy_local) / r_cells, 1.0)))
            vals = defn.height * cross_profile
            b = np.maximum(b, vals * mask)

    max_bed = config.h0 * 0.3
    b = np.minimum(b, max_bed)
    return b.astype(np.float32, copy=False)


# ------------------------------------------------------------------ #
#  VTK Rock meshes                                                   #
# ------------------------------------------------------------------ #
def create_rock_mesh(
    obs: PlacedObstacle, warp_scale: float = 1.0, seed: int | None = None
) -> vtk.vtkPolyData:
    """Return a mildly irregular rock mesh with a natural silhouette."""
    _ = warp_scale
    _ = seed
    r = max(obs.definition.radius - 1.5 * SimConfig.dx, 3.0 * SimConfig.dx)
    params = _rock_params(obs)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(r)
    sphere.SetCenter(0, 0, 0)
    sphere.SetPhiResolution(28)
    sphere.SetThetaResolution(28)
    sphere.Update()

    poly = vtk.vtkPolyData()
    poly.DeepCopy(sphere.GetOutput())
    pts = poly.GetPoints()

    for k in range(pts.GetNumberOfPoints()):
        x, y, z = pts.GetPoint(k)
        d = math.sqrt(x * x + y * y + z * z)
        if d < 1.0e-8:
            continue

        theta = math.atan2(y, x)
        phi = math.acos(max(-1.0, min(1.0, z / d)))

        scale = (
            1.0
            + 0.08 * math.sin(4.0 * theta + params["phase1"]) * math.sin(2.0 * phi + params["phase2"])
            + 0.04 * math.cos(7.0 * theta - 3.0 * phi + params["phase3"])
            + 0.02 * math.sin(11.0 * theta + params["phase4"])
        )
        scale = max(0.90, min(1.12, scale))

        x *= scale
        y *= scale
        z *= scale

        if z < -0.20 * r:
            z *= 0.82

        pts.SetPoint(k, x, y, z)

    poly.Modified()

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputData(poly)
    smooth.SetNumberOfIterations(12)
    smooth.SetRelaxationFactor(0.08)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.SetFeatureAngle(55.0)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()

    transform = vtk.vtkTransform()
    transform.Translate(obs.x, obs.y, r * 0.24 + SimConfig.h0)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(normals.GetOutputPort())
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()


# ------------------------------------------------------------------ #
#  VTK log meshes                                                    #
# ------------------------------------------------------------------ #
def create_log_mesh(obs: PlacedObstacle, warp_scale: float) -> vtk.vtkPolyData:
    """Return a vtkPolyData cylinder positioned on the river bed."""
    _ = warp_scale
    defn = obs.definition

    cyl = vtk.vtkCylinderSource()
    cyl.SetRadius(defn.radius + SimConfig.log_buffer_cells * SimConfig.dx)
    cyl.SetHeight(defn.length + SimConfig.log_buffer_cells * SimConfig.dy)
    cyl.SetResolution(16)
    cyl.Update()

    transform = vtk.vtkTransform()
    transform.Translate(obs.x, obs.y, defn.radius * 0.5 + SimConfig.h0)
    transform.RotateZ(defn.angle)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(cyl.GetOutputPort())
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()


def create_obstacle_actor(obs: PlacedObstacle, warp_scale: float) -> vtk.vtkActor:
    """Create a VTK actor for one placed obstacle."""
    if obs.definition.kind == "rock":
        mesh = create_rock_mesh(obs, warp_scale, seed=_rock_seed(obs))
    else:
        mesh = create_log_mesh(obs, warp_scale)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()

    if obs.definition.kind == "rock":
        prop.SetColor(0.43, 0.39, 0.34)
        prop.SetAmbient(0.18)
        prop.SetDiffuse(0.86)
        prop.SetSpecular(0.06)
        prop.SetSpecularPower(8.0)
    else:
        prop.SetColor(0.55, 0.35, 0.17)
        prop.SetAmbient(0.15)
        prop.SetDiffuse(0.85)
        prop.SetSpecular(0.10)
        prop.SetSpecularPower(10.0)

    return actor