import os
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from config import SimConfig


def export_frame(frame_data: dict, config: SimConfig,
                 frame_idx: int, output_dir: str) -> str:
    """Write one frame as a .vti file and return the path."""
    nx, ny = config.nx, config.ny

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, 1)
    image.SetSpacing(config.dx, config.dy, 1.0)
    image.SetOrigin(0, 0, 0)

    # scalar arrays
    for name in ("h", "eta", "vx", "vy", "speed", "vorticity", "pressure"):
        arr = numpy_to_vtk(frame_data[name].flatten(order="F").astype(np.float32),
                           deep=True)
        arr.SetName(name)
        image.GetPointData().AddArray(arr)

    # velocity vector array  (vx, vy, 0)
    vx = frame_data["vx"].flatten(order="F").astype(np.float32)
    vy = frame_data["vy"].flatten(order="F").astype(np.float32)
    vz = np.zeros_like(vx)
    vec = numpy_to_vtk(np.column_stack([vx, vy, vz]), deep=True)
    vec.SetName("velocity")
    image.GetPointData().AddArray(vec)

    image.GetPointData().SetActiveScalars("h")
    image.GetPointData().SetActiveVectors("velocity")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame_idx:04d}.vti")

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(image)
    writer.SetCompressorTypeToZLib()
    writer.Write()
    return path