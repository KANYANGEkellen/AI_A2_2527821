"""One command that regenerates data, validates, cleans, trains and writes every artifact."""
import argparse
import json
import sys

import pandas as pd

from src import config as C
from src.data_generation import write_dataset, write_fingerprint
from src.modeling import train_all
from src.validation import validate_dataset, validate_seed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service - full pipeline")
    ap.add_argument("--student-id", required=True, help=" ug2527821@ines.ac.rw")
    ap.add_argument("--seed", type=int, required=True, help="LLLL = 7821")
    ap.add_argument("--reg-digits", default="REGDIGITS", help="2527821")
    a = ap.parse_args(argv)
    try:
        seed = validate_seed(a.student_id, a.seed)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    code = C.verification_code(seed)
    print(f"Student: {a.student_id} | Seed: {seed} | Verification code: {code}")

    C.make_dirs()
    csv = write_dataset(seed, C.DATA_DIR / f"AI_A2_{a.reg_digits}.csv")
    digest = write_fingerprint(csv, seed)
    print(f"[1/4] Dataset written: {csv.name}\n      SHA-256: {digest}")

    raw = pd.read_csv(csv)
    quality = validate_dataset(raw)
    clean = raw.drop_duplicates()          # row index kept as row_id; imputation happens later, in the pipeline
    quality.update({"rows_after_dedup": int(len(clean)), "verification_code": code, "sha256": digest})
    (C.ARTIFACTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
    print(f"[2/4] Validation OK: {quality}")

    meta = train_all(clean, seed, code, {"student_id": a.student_id, "sha256": digest,
                                         "rows_raw": quality["rows_raw"],
                                         "rows_clean": quality["rows_after_dedup"]})
    print(f"[3/4] Models trained | regression: {meta['selected_regression']} | "
          f"classifier: {meta['selected_classifier']} | K = {meta['selected_k']}")
    print("[4/4] Artifacts generated:")
    for p in sorted(C.ARTIFACTS_DIR.rglob("*")):
        if p.is_file():
            print("      ", p.relative_to(C.ROOT))
    for p in sorted(C.MODELS_DIR.glob("*")):
        print("      ", p.relative_to(C.ROOT))
    print(f"Done. Verification code: {code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
