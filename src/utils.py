from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_config() -> dict:
    with open(ROOT / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def ensure_dirs() -> None:
    for path in [ROOT/"data"/"processed", ROOT/"models", ROOT/"results"]:
        path.mkdir(parents=True, exist_ok=True)
