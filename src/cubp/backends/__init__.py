from typing import Type

from cubp.args import Backend
from cubp.backends.base import BaseBackend
from cubp.backends.numpy.impl import NumpyBackend

BACKENDS: dict[Backend, Type[BaseBackend]] = {Backend.numpy: NumpyBackend}
