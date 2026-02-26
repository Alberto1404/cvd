from cvd.common.enums import ColourVisionDeficiencies, SimulationAuthors, ColourSpaces
from abc import ABC, abstractmethod
import numpy as np

class ColourBlindnessSimulation(ABC):
    """
    Abstract base class for colour blindness simulation authors

    :param author: The author of the simulation method.
    :type author: SimulationAuthors
    :param colour_deficiency: The type of colour vision deficiency to simulate.
    :type colour_deficiency: ColourVisionDeficiencies
    :param colour_space: The colour space in which the simulation operates.
    :type colour_space: ColourSpaces
    """

    def __init__(self, author: SimulationAuthors, colour_deficiency: ColourVisionDeficiencies, colour_space: ColourSpaces) -> None:
        self._author = author
        self._colour_deficiency = colour_deficiency
        self._colour_space = colour_space

    def get_author(self):
        return self._author

    def get_colour_deficiency(self):
        return self._colour_deficiency

    def get_colour_space(self):
        return self._colour_space

    @abstractmethod
    def simulate(self, image: np.ndarray) -> np.ndarray:
        """
        Abstract method to simulate colour blindness on an input image. Must be implemented by subclasses.

        :param image: Input image as a numpy array.
        :type image: np.ndarray
        :return: Simulated image as a numpy array.
        :rtype: np.ndarray
        """
        raise NotImplementedError()
