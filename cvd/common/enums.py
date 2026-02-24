import enum

class ColourVisionDeficiencies(enum.Enum):
    PROTANOPIA = "protanopia"
    DEUTERANOPIA = "deuteranopia"
    TRITANOPIA = "tritanopia"

class SimulationAuthors(enum.Enum):
    MELILLO = "melillo"
    VIENOT = "vienot"
    MACHADO = "machado"

class RecoloringAuthors(enum.Enum):
    MELILLO = "melillo"
    JEONG = "jeong"

class ColourSpaces(enum.Enum):
    RGB = "rgb"
    LMS = "lms"
