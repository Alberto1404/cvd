from cvd.simulation.base import ColourBlindnessSimulation
from cvd.simulation.authors.machado import MachadoSimulation
from cvd.simulation.authors.vienot import VienotSimulation
from cvd.simulation.authors.melillo import MelilloSimulation
from cvd.common.enums import SimulationAuthors, ColourVisionDeficiencies

class SimulationFactory:
    
    __SIMULATION_CLASSES = {
        SimulationAuthors.MACHADO: MachadoSimulation,
        SimulationAuthors.VIENOT: VienotSimulation,
        SimulationAuthors.MELILLO: MelilloSimulation
    }

    @staticmethod
    def define_simulator(author: SimulationAuthors, colour_deficiency: ColourVisionDeficiencies) -> ColourBlindnessSimulation:
        if author not in SimulationFactory.__SIMULATION_CLASSES:
            raise NotImplementedError(f"Unsupported simulation author: {author}")

        simulation_class = SimulationFactory.__SIMULATION_CLASSES[author]
        return simulation_class(colour_deficiency)