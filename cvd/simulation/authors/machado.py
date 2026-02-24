import numpy as np
from cvd.simulation.base import ColourBlindnessSimulation
from cvd.common.enums import SimulationAuthors, ColourVisionDeficiencies, ColourSpaces


class MachadoSimulation(ColourBlindnessSimulation):

    __DEFICIENCY_MATRICES = {
        ColourVisionDeficiencies.PROTANOPIA: np.array([[0.152286, 1.052583, -0.204868],
                                                       [0.114503, 0.786281, 0.099216],
                                                       [-0.003882, -0.048116,  1.051998]]),

        ColourVisionDeficiencies.DEUTERANOPIA: np.array([[0.367322, 0.860646, -0.227968],
                                                         [0.280085, 0.672501, 0.047413],
                                                         [-0.01182, 0.04294, 0.968881]]),

        ColourVisionDeficiencies.TRITANOPIA: np.array([[1.255528, -0.076749, -0.178779],
                                                       [-0.078411, 0.930809, 0.147602],
                                                       [0.004733, 0.691367, 0.3039]])
    }

    def __init__(self, colour_deficiency: ColourVisionDeficiencies):
        if colour_deficiency not in self.__DEFICIENCY_MATRICES:
            raise NotImplementedError(f"Unsupported colour deficiency: {colour_deficiency}")

        super().__init__(author=SimulationAuthors.MELILLO, colour_deficiency=colour_deficiency, colour_space=ColourSpaces.RGB)

    def simulate(self, image: np.ndarray) -> np.ndarray:
        pixels = image.reshape(-1, 3)
        matrix = self.__DEFICIENCY_MATRICES[self.get_colour_deficiency()]
        simulated_pixels = np.dot(pixels, matrix.T)
        simulated_image = simulated_pixels.reshape(image.shape)
        return simulated_image