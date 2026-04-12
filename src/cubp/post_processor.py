from pathlib import Path

import numpy as np
from PIL import Image
from sarpy.visualization.remap import Density


def post_process(image: np.ndarray, output_path: Path) -> None:
    """
    Takes in a numpy array of a complex image, converts it to be real valued, and
    saves it at the provided output path.
    """

    transform = Density()
    image = transform(image)

    pil_img = Image.fromarray(image)
    pil_img.save(output_path)
