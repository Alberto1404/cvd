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
    
    _RGB_TO_LMS = np.array([[17.8824, 43.5161, 4.11935],
                            [3.45565, 27.1554, 3.86714],
                            [0.0299566, 0.184309, 1.46709]])
    
    _LMS_TO_RGB = np.array([[0.080944, -0.130504, 0.116721],
                            [-0.0102485, 0.0540194, -0.113615],
                            [-0.000365294, -0.00412163, 0.693513]])

    def __init__(self, colour_deficiency: ColourVisionDeficiencies):
        if colour_deficiency not in self.__DEFICIENCY_MATRICES:
            raise NotImplementedError(f"Unsupported colour deficiency: {colour_deficiency}")

        super().__init__(author=SimulationAuthors.MELILLO,
                         colour_deficiency=colour_deficiency,
                         colour_space=ColourSpaces.LMS)

    def _get_deficiency_matrix(self) -> np.ndarray:
        return self.__DEFICIENCY_MATRICES[self.get_colour_deficiency()]

    def _to_space(self) -> np.ndarray:
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]])

    def _from_space(self) -> np.ndarray:
        return np.linalg.inv(self._to_space())
