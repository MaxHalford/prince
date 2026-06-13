from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import datasets
from .ca import CA
from .famd import FAMD
from .gpa import GPA
from .mca import MCA
from .mfa import MFA
from .pca import PCA
from .pga import PGA

try:
    __version__ = version("prince")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["CA", "FAMD", "MCA", "MFA", "PCA", "PGA", "GPA", "__version__", "datasets"]
