"""Deterministic command-line entry point for raw-to-submission SOLAR training."""

from __future__ import annotations

import os
import sys


DETERMINISTIC_ENVIRONMENT = {
    "PYTHONHASHSEED": "42",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def bootstrap_determinism() -> None:
    """Restart once so interpreter-level determinism settings take effect."""
    if all(
        os.environ.get(key) == value
        for key, value in DETERMINISTIC_ENVIRONMENT.items()
    ):
        return
    environment = os.environ.copy()
    environment.update(DETERMINISTIC_ENVIRONMENT)
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def main() -> None:
    bootstrap_determinism()
    from solar_pipeline import main as solar_main

    solar_main()


if __name__ == "__main__":
    main()
