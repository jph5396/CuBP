from cubp.args import (
    CuBPArguments, 
    ImageBounds, 
    TargetLonLat
)

from cubp.coordinates import build_enu_image_grid, convert_enu_grid_to_ecef

import numpy as np
from sarpy.io.phase_history.cphd import CPHDTypeReader
from sarpy.io.phase_history.converter import open_phase_history


def _prepare_coordinate_grid(
        reader: CPHDTypeReader,
        image_bounds: ImageBounds,
        spacing: float,
        target: TargetLonLat | None = None, 
        
                
) -> np.ndarray:

    ref_point = reader.cphd_meta.ReferenceGeometry.SRP.ECF # pyright: ignore 
    ref_vec = np.array(
        [
            ref_point.X,
            ref_point.Y,
            ref_point.Z
        ]
    )

    t = None
    if target is not None: 
        t = (target.lat, target.lon) 


    enu_grid = build_enu_image_grid(
        image_bounds.x, 
        image_bounds.y, 
        spacing, 
        ref_vec, 
        t
    )

    ecef_grid = convert_enu_grid_to_ecef(
        enu_grid, 
        ref_vec
    )

    return ecef_grid



def main() -> None: 
    args = CuBPArguments()  # pyright: ignore[reportCallIssue]
    
    reader = open_phase_history(args.cphd_file)
    ecef_grid = _prepare_coordinate_grid(
        reader, 
        args.image_bounds, 
        args.image_spacing,
        args.target_lon_lat, 
    )


