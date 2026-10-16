"""Load saved artifacts and make one prediction (used by predict.py and app.py)."""
import json

import joblib
import pandas as pd

from src import config as C

FILES = ["regression_model.joblib", "classification_model.joblib",
         "cluster_model.joblib", "run_meta.json"]


class Predictor:
    """Holds the saved models; raises a helpful error if training has not been run."""

    def __init__(self):
        missing = [f for f in FILES if not (C.MODELS_DIR / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing model files {missing} in {C.MODELS_DIR}. "
                                    f"Run: python run_all.py --student-id <id> --seed <LLLL>")
        self.reg = joblib.load(C.MODELS_DIR / "regression_model.joblib")
        self.clf = joblib.load(C.MODELS_DIR / "classification_model.joblib")
        self.cluster = joblib.load(C.MODELS_DIR / "cluster_model.joblib")
        self.meta = json.loads((C.MODELS_DIR / "run_meta.json").read_text())

    def predict(self, record: dict) -> dict:
        """Return yield, risk, probability, cluster and verification code for one valid record."""
        X = pd.DataFrame([record])[C.FEATURES]
        prob = float(self.clf.predict_proba(X)[0, 1])
        cl = int(self.cluster.predict(X)[0])
        return {
            "yield_estimate_tons": round(float(self.reg.predict(X)[0]), 3),
            "high_risk_prediction": bool(prob >= 0.5),
            "high_risk_probability": round(prob, 4),
            "cluster": cl,
            "cluster_profile": self.meta["cluster_profiles"].get(str(cl), ""),
            "verification_code": self.meta["verification_code"],
            "disclaimer": C.DISCLAIMER,
        }
