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
#  Bed-elevation builder (numpy, vectorized)                          #
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
            # Smooth cosine bump (zero-gradient at center and edge)
            r_cells = defn.radius / config.dx
            dist = np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)
            mask = dist < r_cells
            vals = defn.height * 0.5 * (1.0 + np.cos(np.pi * np.minimum(dist / r_cells, 1.0)))
            b = np.maximum(b, vals * mask)

        elif defn.kind == "log":
            # Smooth cosine ridge
            r_cells = defn.radius / config.dx
            length_cells = defn.length / config.dx
            angle_rad = math.radians(defn.angle)
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

    # Cap bed so water depth is always >= 30% of h0 (prevents extreme acceleration)
    max_bed = config.h0 * 0.7
    b = np.minimum(b, max_bed)
    return b


# ------------------------------------------------------------------ #
#  VTK Rock meshes                                                #
# ------------------------------------------------------------------ #
def create_rock_mesh(obs: PlacedObstacle) -> vtk.vtkPolyData:
    """Return a vtkPolyData sphere with surface noise (rocky look)."""
    rng = random.Random()
    r = obs.definition.radius

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(r)
    sphere.SetCenter(0, 0, 0)
    sphere.SetPhiResolution(16)
    sphere.SetThetaResolution(16)
    sphere.Update()

    poly = vtk.vtkPolyData()
    poly.DeepCopy(sphere.GetOutput())
    pts = poly.GetPoints()

    for k in range(pts.GetNumberOfPoints()):
        p = list(pts.GetPoint(k))
        d = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
        if d > 0:
            noise = r * 0.15 * (rng.random() - 0.5)
            factor = (d + noise) / d
            p = [c * factor for c in p]
        pts.SetPoint(k, *p)

    # Transform to world position: place bottom hemisphere at z = 0
    transform = vtk.vtkTransform()
    transform.Translate(obs.x, obs.y, r * 0.3)  # slightly embedded
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()

# ------------------------------------------------------------------ #
#  VTK log meshes                                                #
# ------------------------------------------------------------------ #
def create_log_mesh(obs: PlacedObstacle, warp_scale: float) -> vtk.vtkPolyData:
    """Return a vtkPolyData cylinder positioned on the river bed."""
    defn = obs.definition
    cyl = vtk.vtkCylinderSource()
    cyl.SetRadius(defn.radius)
    cyl.SetHeight(defn.length)
    cyl.SetResolution(16)
    cyl.Update()

    transform = vtk.vtkTransform()
    transform.Translate(obs.x, obs.y, defn.radius * 0.5)
    transform.RotateZ(defn.angle)       # rotate in xy-plane
    transform.RotateX(90)               # lay cylinder on its side

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(cyl.GetOutputPort())
    tf.SetTransform(transform)
    tf.Update()
    return tf.GetOutput()


def create_obstacle_actor(obs: PlacedObstacle, warp_scale: float) -> vtk.vtkActor:
    """Create a VTK actor for one placed obstacle."""
    if obs.definition.kind == "rock":
        mesh = create_rock_mesh(obs, warp_scale, seed=hash(f"{obs.x}_{obs.y}") % 2**31)
    else:
        mesh = create_log_mesh(obs, warp_scale)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if obs.definition.kind == "rock":
        actor.GetProperty().SetColor(0.45, 0.40, 0.35)   # grey-brown
    else:
        actor.GetProperty().SetColor(0.55, 0.35, 0.17)   # wood brown
    actor.GetProperty().SetSpecular(0.1)
    return actor
