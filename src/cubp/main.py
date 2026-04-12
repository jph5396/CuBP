import logging
import sys
import time
from typing import Type

from pythonjsonlogger.json import JsonFormatter
from sarpy.io.phase_history.converter import open_phase_history

from cubp.args import CuBPArguments
from cubp.backends import BACKENDS, BaseBackend
from cubp.post_processor import post_process


def _configure_logger(args: CuBPArguments) -> logging.Logger:
    """
    Some basic logging configuration.
    """
    logger = logging.getLogger(__name__)
    formatter = JsonFormatter()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.handlers = [stream_handler]
    if args.log_file is not None:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    return logger


def main() -> None:
    args = CuBPArguments()  # pyright: ignore[reportCallIssue]
    logger = _configure_logger(args)

    logger.debug({"message": "configuration", **args.model_dump()})
    backend_class: Type[BaseBackend] | None = BACKENDS.get(args.backend)
    if backend_class is None:
        logger.critical(f"backend: {args.backend} is not yet implemented")
        exit(1)

    logger.debug(f"loaded backend {args.backend}")

    backend: BaseBackend = backend_class(
        cphd_reader=open_phase_history(str(args.cphd_file)),
        image_bounds=args.image_bounds,
        spacing=args.image_spacing,
        pulse_limit=args.pulse_limit,
    )

    logger.debug("generating grid")
    start = time.perf_counter()
    backend.create_grid()
    grid_complete_t = time.perf_counter()
    logger.debug({"grid_gen_time": grid_complete_t - start})

    logger.debug({"message": "forming image"})
    start_formation = time.perf_counter()
    image_mat = backend.form_image()
    complete_formation = time.perf_counter()
    logger.debug({"image_formation_time": complete_formation - start_formation})

    post_process(image_mat, args.output_file)
    image_saved_t = time.perf_counter()
    logger.debug({"message": f"saved file to {str(args.output_file)}", "total_time": image_saved_t - start})


if __name__ == "__main__":
    main()
