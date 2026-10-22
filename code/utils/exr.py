from pathlib import Path
import numpy as np

def read_exr_red(path):
    import Imath
    import OpenEXR

    path = Path(path)
    exr = OpenEXR.InputFile(str(path))
    try:
        window = exr.header()['dataWindow']
        if window.min.x != 0 or window.min.y != 0:
            raise ValueError(f"{path}: expected an EXR data window starting at (0, 0)")
        height = window.max.y - window.min.y + 1
        width = window.max.x - window.min.x + 1
        red = exr.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(red, dtype=np.float32).reshape(height, width).copy()
    finally:
        exr.close()
