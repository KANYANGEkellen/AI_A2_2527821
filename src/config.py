"""Project paths, feature lists and shared constants (single source of truth)."""
import os
from pathlib import Path

# Outputs go to ROOT. Tests set AI_A2_ROOT so they never overwrite real artifacts.
ROOT = Path(os.environ.get("AI_A2_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

NUMERIC = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha"]
CATEGORICAL = ["season"]
BINARY = ["cooperative_support"]
FEATURES = NUMERIC + CATEGORICAL + BINARY
TARGET_REG = "yield_tons"
TARGET_CLF = "high_risk"
FORBIDDEN_FEATURES = {"yield_tons", "yield_per_ha", "high_risk"}

RANGES = {
    "rainfall_mm": (600, 1400),
    "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80),
    "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0),
}

DISCLAIMER = ("Synthetic classroom prototype. Decision support only: it does not replace an "
              "agronomist and must not decide finance, insurance or public services. "
              "Human review is required.")


def verification_code(seed: int) -> str:
    """Return the individual verification code AI-A2-LLLL."""
    return f"AI-A2-{seed:04d}"


def make_dirs() -> None:
    """Create output folders if they do not exist."""
    for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
