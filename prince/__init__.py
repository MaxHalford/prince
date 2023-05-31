import importlib.metadata
from .ca import CA
from .famd import FAMD
from .mca import MCA
from .mfa import MFA
from .pca import PCA
from .gpa import GPA
from . import datasets

__version__ = importlib.metadata.version("prince")
