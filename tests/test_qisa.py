import random
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when tests are run via `pytest`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer.qisa import AnnealConfig, anneal


def test_qisa_finds_low_value_on_quadratic():
    # Minimize f(x) = (x-3)^2 over integers with neighbor moves +/- 1.
    def neighbor(x: int, rng: random.Random) -> int:
        return x + (1 if rng.random() < 0.5 else -1)

    def energy(x: int) -> float:
        return float((x - 3) ** 2)

    result = anneal(
        initial_state=50,
        neighbor=neighbor,
        energy=energy,
        config=AnnealConfig(steps=2000, t_start=5.0, t_end=0.01, gamma=1.5, reheats=1, seed=1),
    )

    assert result.best_energy == 0.0
    assert result.best_state == 3
