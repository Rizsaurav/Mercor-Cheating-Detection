"""
src/utils/config.py
-------------------
Loads cost_config.yaml and exposes typed constants project-wide.
All other modules import from here — never hardcode costs/paths.
"""
from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parents[2]   # cheating/  (src/utils/config.py → parents[2])
_CFG_PATH = _ROOT / "configs" / "cost_config.yaml"


def _load() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load()

# ── Cost constants ────────────────────────────────────────────────────────────
COST_FN: int = _cfg["costs"]["false_negative"]          # 600
COST_FP_BLOCK: int = _cfg["costs"]["fp_auto_block"]     # 300
COST_FP_REVIEW: int = _cfg["costs"]["fp_manual_review"] # 150
COST_TP_REVIEW: int = _cfg["costs"]["tp_manual_review"] # 5
COST_RATIO: float = _cfg["cost_ratio"]                  # 2.0

# ── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {k: _ROOT / v for k, v in _cfg["paths"].items()}

# ── CV settings ───────────────────────────────────────────────────────────────
CV = _cfg["cv"]

# ── Threshold grid ────────────────────────────────────────────────────────────
THRESHOLD_GRID = _cfg["threshold_grid"]

# ── Graph settings ────────────────────────────────────────────────────────────
GRAPH_CFG = _cfg["graph"]

# ── Pseudo-label settings ─────────────────────────────────────────────────────
PSEUDO_CFG = _cfg["pseudo_label"]
