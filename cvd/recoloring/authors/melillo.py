import numpy as np
from cvd.recoloring.base import ColourBlindnessRecoloring
from cvd.simulation.base import ColourBlindnessSimulation
from cvd.common.enums import RecoloringAuthors, ColourVisionDeficiencies

class MelilloRecoloring(ColourBlindnessRecoloring):
    """
    Implementation of the recoloring method proposed by Melillo et al. for recoloring colour blindness.

    Algorithm: 
    1. Simulate the deficiency 
    2. Compute the error image
    3. Apply correction to hte error
    4. Add the correction to the simulated image
    """

    _CORRECTION_MATRICES = {
        ColourVisionDeficiencies.PROTANOPIA: np.array([[0, 0, 0],
                                                       [0.5, 1, 0],
                                                       [0.5, 0, 1]]),

        ColourVisionDeficiencies.DEUTERANOPIA: np.array([[1, 0.5, 0],
                                                         [0, 0, 0],
                                                         [0, 0.5, 1]]),

        ColourVisionDeficiencies.TRITANOPIA: np.array([[1, 0, 0.7],
                                                        [0, 1, 0.7],
                                                        [0, 0, 0]])
    }

    def __init__(self, colour_deficiency: ColourVisionDeficiencies, simulator: ColourBlindnessSimulation):
        super().__init__(author=RecoloringAuthors.MELILLO, colour_deficiency=colour_deficiency, simulator=simulator)
        self._correction_matrix = self._CORRECTION_MATRICES[colour_deficiency]

    def recolor(self, image: np.ndarray) -> np.ndarray:
        simulated_image = self.get_simulator().simulate(image)
        error_image = image - simulated_image
        
        error_pixels = error_image.reshape(-1, 3)
        correction_pixels = np.dot(error_pixels, self._correction_matrix.T)
        correction = correction_pixels.reshape(image.shape)
        recolored = simulated_image + correction

        return recolored
