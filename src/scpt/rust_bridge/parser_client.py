"""Re-exports from the `pcb_parser` PyO3 wheel.

No logic here — this module is a boundary. If you need to transform the
parsed design or cache results, do that in `env/` or `training/`.
"""
from pcb_parser import (
    load_kicad_pcb,
    hpwl,
    hpwl_incremental,
    clearance_cost,
)

__all__ = [
    "load_kicad_pcb",
    "hpwl",
    "hpwl_incremental",
    "clearance_cost",
]
