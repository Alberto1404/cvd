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

    _RGB_TO_LMS = np.array([[17.8824, 43.5161, 4.11935],
                            [3.45565, 27.1554, 3.86714],
                            [0.0299566, 0.184309, 1.46709]])
    
    _LMS_TO_RGB = np.array([[0.080944, -0.130504, 0.116721],
                            [-0.0102485, 0.0540194, -0.113615],
                            [-0.000365294, -0.00412163, 0.693513]])

    def __init__(self, colour_deficiency: ColourVisionDeficiencies):
        if colour_deficiency not in self.__DEFICIENCY_MATRICES:
            raise NotImplementedError(f"Unsupported colour deficiency: {colour_deficiency}")

        super().__init__(author=SimulationAuthors.VIENOT, colour_deficiency=colour_deficiency, colour_space=ColourSpaces.LMS)

    def simulate(self, image: np.ndarray) -> np.ndarray:
        c = self._LMS_TO_RGB @ self.__DEFICIENCY_MATRICES[self.get_colour_deficiency()] @ self._RGB_TO_LMS
        
        pixels = image.reshape(-1, 3)
        simulated_pixels = np.dot(pixels, c.T)
        simulated_image = simulated_pixels.reshape(image.shape)
        
        return simulated_image
