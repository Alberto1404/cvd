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

    def simulate(self, image: np.ndarray) -> np.ndarray:
        """
        Template method to apply the colour blindness simulation to the input image.

        :param image: Input image as a numpy array.
        :type image: np.ndarray
        :return: Simulated image as a numpy array.
        :rtype: np.ndarray
        """

        c = self._from_space() @ self._get_deficiency_matrix() @ self._to_space()

        pixels = image.reshape(-1, 3)
        simulated_pixels = pixels @ c.T
        simulated_image = simulated_pixels.reshape(image.shape)

        return simulated_image

    @abstractmethod
    def _get_deficiency_matrix(self) -> np.ndarray:
        """
        Abstract method to get the deficiency matrix specific to the simulation instance.

        :return: Deficiency matrix as a numpy array.
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def _from_space() -> np.ndarray:
        """
        Method to convert the input image from RGB to the simulation's colour space. Override by authors that
        operate in a different colour space.

        :return: Converted image as a numpy array.
        :rtype: np.ndarray
        """
        return np.eye(3)

    def _to_space() -> np.ndarray:
        """
        Method to convert the simulated image back to RGB from the simulation's colour space. Override by authors that
        operate in a different colour space.

        :return: Converted image as a numpy array.
        :rtype: np.ndarray
        """
        return np.eye(3)
