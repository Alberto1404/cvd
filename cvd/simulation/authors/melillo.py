import numpy as np
from cvd.simulation.base import ColourBlindnessSimulation
from cvd.common.enums import SimulationAuthors, ColourVisionDeficiencies, ColourSpaces


class MelilloSimulation(ColourBlindnessSimulation):

    __DEFICIENCY_MATRICES = {
        ColourVisionDeficiencies.PROTANOPIA: np.array([[0, 2.02344, -2.52581],
                                                       [0, 1, 0],
                                                       [0, 0, 1]]),

        ColourVisionDeficiencies.DEUTERANOPIA: np.array([[1.42319, -0.88995, 1.77557],
                                                         [0.67558, -0.42203, 2.82788],
                                                         [0.00267, -0.00504, 0.99914]]),

        ColourVisionDeficiencies.TRITANOPIA: np.array([[0.95451, -0.04719, 2.74872],
                                                       [-0.00447, 0.96543, 0.88835],
                                                       [-0.01251, 0.07312, -0.01161]])
    }

    def __init__(self, colour_deficiency: ColourVisionDeficiencies):
        if colour_deficiency not in self.__DEFICIENCY_MATRICES:
            raise NotImplementedError(f"Unsupported colour deficiency: {colour_deficiency}")

        super().__init__(author=SimulationAuthors.MELILLO, colour_deficiency=colour_deficiency, colour_space=ColourSpaces.LMS)

    def simulate(self, image: np.ndarray) -> np.ndarray:
        pass