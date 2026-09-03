"""
Repository-anchored paths.

Every module used to write to the bare relative string "models/saved", which
resolves against the current working directory. The pipeline therefore only
worked when launched from the repository root: a teammate running
`pytest tests/` or the pipeline from anywhere else scattered checkpoints into
whatever directory they happened to be in, or failed outright.

Paths here are derived from this file's own location, so they are correct
regardless of where the process was started.
"""
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
MODELS_DIR  = REPO_ROOT / "models" / "saved"
DATA_DIR    = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def models_dir() -> str:
    """
    Return the model checkpoint directory, creating it if absent.

    Returns:
        Absolute path to models/saved as a string.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR)
