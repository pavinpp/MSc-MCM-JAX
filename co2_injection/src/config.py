from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    NX: int = 128
    NY: int = 64
    STEPS: int = 2000
    RHO_BRINE: float = 1.0
    RHO_CO2: float = 0.8
    TAU_CO2: float = 0.7
    D_SALT: float = 0.01
    K_SP: float = 1.0
    INLET_Y_START: int = 0
    INLET_Y_END: int = 32
    LEARNING_RATE: float = 0.05
