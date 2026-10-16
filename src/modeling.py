"""Train, compare and save regression, classification and clustering models."""
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config as C
from src.clustering import run_clustering
from src.plots import plot_clusters, plot_confusion, plot_residuals
from src.preprocessing import build_preprocessor
from src.validation import assert_no_leakage


def _dump(obj, path):
    path.write_text(json.dumps(obj, indent=2))


def _pipe(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def train_all(df: pd.DataFrame, seed: int, code: str, extra_meta: dict) -> dict:
    """Run the full modelling stage on the CLEANED dataframe and write every artifact."""
    C.make_dirs()
    X = df[C.FEATURES]
    assert_no_leakage(X.columns)
    # One fixed split (random_state = LLLL) shared by all three tasks.
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=seed,
                                      stratify=df[C.TARGET_CLF])
    Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
    ytr_r, yte_r = df.loc[idx_tr, C.TARGET_REG], df.loc[idx_te, C.TARGET_REG]
    ytr_c, yte_c = df.loc[idx_tr, C.TARGET_CLF], df.loc[idx_te, C.TARGET_CLF]

    # ---------- Regression ----------
    cv = KFold(5, shuffle=True, random_state=seed)
    reg_models = {"mean_baseline": DummyRegressor(strategy="mean"),
                  "linear_regression": LinearRegression(),
                  "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed)}
    reg_res, reg_fit = {}, {}
    for name, m in reg_models.items():
        p = _pipe(m)
        cv_rmse = -cross_val_score(p, Xtr, ytr_r, cv=cv, scoring="neg_root_mean_squared_error").mean()
        p.fit(Xtr, ytr_r)
        pred = p.predict(Xte)
        reg_res[name] = {"cv_rmse_train": round(float(cv_rmse), 4),
                         "mae": round(float(mean_absolute_error(yte_r, pred)), 4),
                         "rmse": round(float(np.sqrt(mean_squared_error(yte_r, pred))), 4),
                         "r2": round(float(r2_score(yte_r, pred)), 4)}
        reg_fit[name] = p
    best_reg = min((n for n in reg_res if n != "mean_baseline"), key=lambda n: reg_res[n]["cv_rmse_train"])
    _dump({"verification_code": code, "models": reg_res, "selected_model": best_reg,
           "selection_reason": f"{best_reg} has the lowest 5-fold CV RMSE on the training set "
                               f"({reg_res[best_reg]['cv_rmse_train']}); test set used only for reporting."},
          C.ARTIFACTS_DIR / "regression_metrics.json")
    reg_pred = reg_fit[best_reg].predict(Xte)
    joblib.dump(reg_fit[best_reg], C.MODELS_DIR / "regression_model.joblib")
    plot_residuals(yte_r, reg_pred, C.PLOTS_DIR / "residuals.png", code)

    # ---------- Classification ----------
    scv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf_models = {"majority_baseline": DummyClassifier(strategy="most_frequent"),
                  "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
                  "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed)}
    clf_res, clf_fit = {}, {}
    for name, m in clf_models.items():
        p = _pipe(m)
        cv_f1 = cross_val_score(p, Xtr, ytr_c, cv=scv, scoring="f1").mean()
        p.fit(Xtr, ytr_c)
        pred, proba = p.predict(Xte), p.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte_c, proba)) if yte_c.nunique() == 2 else None
        clf_res[name] = {"cv_f1_train": round(float(cv_f1), 4),
                         "precision": round(float(precision_score(yte_c, pred, zero_division=0)), 4),
                         "recall": round(float(recall_score(yte_c, pred, zero_division=0)), 4),
                         "f1": round(float(f1_score(yte_c, pred, zero_division=0)), 4),
                         "roc_auc": None if auc is None else round(auc, 4),
                         "confusion_matrix": confusion_matrix(yte_c, pred, labels=[0, 1]).tolist()}
        clf_fit[name] = p
    best_clf = max((n for n in clf_res if n != "majority_baseline"), key=lambda n: clf_res[n]["cv_f1_train"])
    _dump({"verification_code": code, "models": clf_res, "selected_model": best_clf,
           "selection_reason": f"{best_clf} has the highest 5-fold CV F1 on the training set "
                               f"({clf_res[best_clf]['cv_f1_train']}). The majority baseline never "
                               f"predicts the positive class when classes are balanced, so its F1 is low."},
          C.ARTIFACTS_DIR / "classification_metrics.json")
    clf_pred = clf_fit[best_clf].predict(Xte)
    clf_proba = clf_fit[best_clf].predict_proba(Xte)[:, 1]
    joblib.dump(clf_fit[best_clf], C.MODELS_DIR / "classification_model.joblib")
    plot_confusion(clf_res[best_clf]["confusion_matrix"], C.PLOTS_DIR / "confusion_matrix.png", code)

    # ---------- Clustering (features only) ----------
    cpipe, crep, c_test, coords, c_train = run_clustering(Xtr, Xte, ytr_r, seed, code)
    _dump(crep, C.ARTIFACTS_DIR / "clustering_report.json")
    joblib.dump(cpipe, C.MODELS_DIR / "cluster_model.joblib")
    plot_clusters(coords, c_train, C.PLOTS_DIR / "clusters.png", code)

    # ---------- Test predictions ----------
    pd.DataFrame({
        "row_id": idx_te, "actual_yield_tons": yte_r.values, "predicted_yield_tons": reg_pred,
        "error_tons": reg_pred - yte_r.values, "abs_error_tons": np.abs(reg_pred - yte_r.values),
        "actual_high_risk": yte_c.values, "predicted_high_risk": clf_pred,
        "probability_high_risk": clf_proba, "cluster": c_test,
    }).to_csv(C.ARTIFACTS_DIR / "test_predictions.csv", index=False, float_format="%.4f")

    meta = {**extra_meta, "verification_code": code, "seed": seed,
            "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
            "selected_regression": best_reg, "selected_classifier": best_clf,
            "selected_k": crep["selected_k"],
            "cluster_profiles": {k: v["summary"] for k, v in crep["profiles"].items()},
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "disclaimer": C.DISCLAIMER}
    _dump(meta, C.MODELS_DIR / "run_meta.json")
    return meta
