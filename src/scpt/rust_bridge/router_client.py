"""Re-exports from the `pcb_router` PyO3 wheel.

No logic here — this module is a boundary. If you need post-routing analysis,
do that in `scripts/evaluate.py`.
"""
from pcb_router import route

__all__ = ["route"]
