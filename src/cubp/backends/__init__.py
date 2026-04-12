from enum import StrEnum


class Backend(StrEnum):
    numpy = "numpy"
    cupy = "cupy"
    cuda = "cuda"


BACKENDS = {}
