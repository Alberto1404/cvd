import numpy as np
from cvd.simulation.base import ColourBlindnessSimulation
from cvd.common.enums import SimulationAuthors, ColourVisionDeficiencies, ColourSpaces


class VienotSimulation(ColourBlindnessSimulation):

    __DEFICIENCY_MATRICES = {
        ColourVisionDeficiencies.PROTANOPIA: np.array([[0, 2.02344, -2.52581],
                                                       [0, 1, 0],
                                                       [0, 0, 1]]),

        ColourVisionDeficiencies.DEUTERANOPIA: np.array([[1, 0, 0],
                                                         [0.494207, 0, 1.24827],
                                                         [0, 0, 1]]),
    }

    def __init__(self, colour_deficiency: ColourVisionDeficiencies):
        if colour_deficiency not in self.__DEFICIENCY_MATRICES:
            raise NotImplementedError(f"Unsupported colour deficiency: {colour_deficiency}")

        super().__init__(author=SimulationAuthors.VIENOT, colour_deficiency=colour_deficiency, colour_space=ColourSpaces.LMS)

    def simulate(self, image: np.ndarray) -> np.ndarray:
        pass