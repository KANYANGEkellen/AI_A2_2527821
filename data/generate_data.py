"""Standalone generator: python data/generate_data.py --seed 4671 --reg-digits 2424671"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, verification_code  # noqa: E402
from src.data_generation import write_dataset, write_fingerprint  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--reg-digits", required=True)
    a = p.parse_args()
    path = write_dataset(a.seed, DATA_DIR / f"AI_A2_{a.reg_digits}.csv")
    print(verification_code(a.seed), path, write_fingerprint(path, a.seed))
