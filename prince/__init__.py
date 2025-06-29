from __future__ import annotations

from . import datasets
from .ca import CA
from .famd import FAMD
from .gpa import GPA
from .mca import MCA
from .mfa import MFA
from .pca import PCA

__all__ = ["CA", "FAMD", "MCA", "MFA", "PCA", "GPA", "datasets"]
