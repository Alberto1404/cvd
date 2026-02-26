from cvd.common.enums import RecoloringAuthors, ColourVisionDeficiencies, SimulationAuthors
from cvd.recoloring.base import ColourBlindnessRecoloring
from cvd.recoloring.authors.melillo import MelilloRecoloring
from cvd.simulation.factory import SimulationFactory

class RecoloringFactory:
    __RECOLORING_CLASSES = {
        RecoloringAuthors.MELILLO: MelilloRecoloring
    }

    @staticmethod
    def define_recoloring(author: RecoloringAuthors, colour_deficiency: ColourVisionDeficiencies,
                          simulation_author: SimulationAuthors) -> ColourBlindnessRecoloring:
        if author not in RecoloringFactory.__RECOLORING_CLASSES:
            raise NotImplementedError(f"Unsupported recoloring author: {author}")

        simulator = SimulationFactory.define_simulator(author=simulation_author, colour_deficiency=colour_deficiency)
        recoloring_class = RecoloringFactory.__RECOLORING_CLASSES[author]

        return recoloring_class(colour_deficiency=colour_deficiency, simulator=simulator)
