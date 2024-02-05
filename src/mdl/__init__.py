# read version from installed package
from __future__ import annotations

from importlib.metadata import version
__version__ = version('mdl')
