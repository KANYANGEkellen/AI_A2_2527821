"""CLI: python predict.py --json samples/valid.json"""
import argparse
import json
import sys
from pathlib import Path

from src.inference import Predictor
from src.validation import validate_record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Predict for one farm record (JSON file).")
    ap.add_argument("--json", required=True, help="Path to a JSON file with one record")
    args = ap.parse_args(argv)
    try:
        record = json.loads(Path(args.json).read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read JSON: {e}", file=sys.stderr)
        return 2
    errors = validate_record(record)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    try:
        print(json.dumps(Predictor().predict(record), indent=2))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
