import numpy as np
import numpy.typing as npt


def _initial_transition(
    k: int, rng: np.random.Generator, diagonal: float
) -> npt.NDArray[np.float64]:
    off = (1.0 - diagonal) / (k - 1)
    p = np.full((k, k), off) + (diagonal - off) * np.eye(k)
    p = p * rng.uniform(0.9, 1.1, size=(k, k))
    return p / p.sum(axis=1, keepdims=True)
