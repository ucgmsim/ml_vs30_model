from abc import ABC, abstractmethod

import numpy as np

from .. import constants


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def get_values(self, coords: np.ndarray, variable: constants.InputVariable) -> np.ndarray:
        """Retrieve values for the given coordinates and variable."""
        raise NotImplementedError("Subclasses must implement this method.")