"""Reproducible synthetic data generator for the Musanze potato cooperative."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, make_dirs, verification_code


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Create the raw dataset (with deliberate missing values and 3 duplicates)."""
    rng = np.random.default_rng(seed)
    n = 240 + (seed % 41)
    df = pd.DataFrame({
        "rainfall_mm": rng.uniform(600, 1400, n),
        "avg_temp_c": rng.uniform(12, 26, n),
        "soil_moisture_pct": rng.uniform(20, 80, n),
        "fertilizer_kg": rng.uniform(0, 250, n),
        "field_size_ha": rng.uniform(0.2, 4.0, n),
        "season": rng.choice(["A", "B"], n),
        "cooperative_support": rng.integers(0, 2, n),
    })
    season_effect = np.where(df["season"] == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, n)
    y = (0.004 * df["rainfall_mm"] + 0.06 * df["soil_moisture_pct"]
         + 0.02 * df["fertilizer_kg"] + 1.30 * df["field_size_ha"]
         - 0.25 * (df["avg_temp_c"] - 18).abs() + 0.70 * df["cooperative_support"]
         + season_effect + noise)
    df["yield_tons"] = np.clip(y, 0.50, None)
    per_ha = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (per_ha < per_ha.median()).astype(int)

    m = round(0.05 * n)
    df.loc[rng.choice(n, m, replace=False), "soil_moisture_pct"] = np.nan
    df.loc[rng.choice(n, m, replace=False), "fertilizer_kg"] = np.nan
    dup_pos = rng.choice(n, 3, replace=False)
    return pd.concat([df, df.iloc[dup_pos]], ignore_index=True)


def write_dataset(seed: int, csv_path: Path) -> Path:
    """Generate the dataset and save it with four decimal places."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    generate_dataframe(seed).to_csv(csv_path, index=False, float_format="%.4f")
    return csv_path


def sha256_file(path: Path) -> str:
    """SHA-256 of the exact CSV bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fingerprint(csv_path: Path, seed: int) -> str:
    """Write artifacts/data_fingerprint.txt and return the hash."""
    make_dirs()
    digest = sha256_file(csv_path)
    text = (f"verification_code: {verification_code(seed)}\n"
            f"file: {Path(csv_path).name}\nsha256: {digest}\n")
    (ARTIFACTS_DIR / "data_fingerprint.txt").write_text(text)
    return digest
