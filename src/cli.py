"""Console entry points (``uv run train ...``).

The modules under ``src/`` import each other as top-level packages (``from data
import ...``) and Hydra resolves ``config_path="configs"`` relative to the
``__main__`` script file, so both only work when ``src/train.py`` runs as the
main script. ``runpy`` reproduces that exactly: ``uv run train X=Y`` behaves
like ``uv run python src/train.py X=Y`` (Hydra reads overrides from
``sys.argv``).
"""

import os
import runpy
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))


def _run_as_main(script: str) -> None:
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    runpy.run_path(os.path.join(_SRC, script), run_name="__main__")


def train() -> None:
    _run_as_main("train.py")
