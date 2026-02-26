from abc import ABC, abstractmethod
import numpy as np

from cvd.common.enums import ColourVisionDeficiencies, RecoloringAuthors
from cvd.simulation.base import ColourBlindnessSimulation


class ColourBlindnessRecoloring(ABC):
    """
    Abstract base class for recoloring authors

    :param author: The author of the recoloring method.
    :type author: RecoloringAuthors
    :param colour_deficiency: The type of colour vision deficiency to simulate.
    :type colour_deficiency: ColourVisionDeficiencies
    """

    def __init__(self, author: RecoloringAuthors, colour_deficiency: ColourVisionDeficiencies, simulator: ColourBlindnessSimulation) -> None:
        self._author = author
        self._colour_deficiency = colour_deficiency
        self._simulator = simulator

    def get_author(self):
        return self._author

    def get_colour_deficiency(self):
        return self._colour_deficiency

    def get_simulator(self):
        return self._simulator

    @abstractmethod
    def recolor(self, image: np.ndarray) -> np.ndarray:
        """
        Abstract method to recolor an input image for a specific colour vision deficiency. Must be implemented by subclasses.

        :param image: Input image as a numpy array.
        :type image: np.ndarray.
        :return: Recolored image as a numpy array.
        :rtype: np.ndarray
        """

        raise NotImplementedError()
