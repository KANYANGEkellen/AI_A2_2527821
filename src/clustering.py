"""KMeans clustering (K=2..6, silhouette-selected) with plain-language profiles. No targets used."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.config import NUMERIC
from src.preprocessing import build_preprocessor

LABELS = {"rainfall_mm": "rainfall", "avg_temp_c": "temperature", "soil_moisture_pct": "soil moisture",
          "fertilizer_kg": "fertilizer", "field_size_ha": "field size"}


def run_clustering(X_train, X_test, yield_train, seed, code):
    """Fit on train features, return (pipeline, report, test_labels, pca_coords, train_labels)."""
    prep = build_preprocessor()
    Z = prep.fit_transform(X_train)
    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        scores[k] = float(silhouette_score(Z, km.labels_))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Z)
    pipe = Pipeline([("prep", prep), ("km", km)])
    tr = X_train.copy()
    tr["cluster"] = km.labels_
    mean, std = X_train[NUMERIC].mean(), X_train[NUMERIC].std()
    profiles = {}
    for c in range(best_k):
        sub = tr[tr["cluster"] == c]
        z = (sub[NUMERIC].mean() - mean) / std
        traits = [f"{'high' if z[f] > 0 else 'low'} {LABELS[f]} ({sub[f].mean():.1f})"
                  for f in NUMERIC if abs(z[f]) > 0.4] or ["close to the average farm"]
        profiles[str(c)] = {
            "n_farms": int(len(sub)),
            "center_original_units": {f: round(float(sub[f].mean()), 3) for f in NUMERIC},
            "share_season_A": round(float((sub["season"] == "A").mean()), 3),
            "share_cooperative_support": round(float(sub["cooperative_support"].mean()), 3),
            "posthoc_mean_yield_tons": round(float(yield_train.loc[sub.index].mean()), 3),
            "summary": f"Cluster {c}: {len(sub)} farms with " + ", ".join(traits) + ".",
        }
    report = {
        "verification_code": code,
        "silhouette_scores": {str(k): round(v, 4) for k, v in scores.items()},
        "selected_k": best_k,
        "selection_reason": f"K={best_k} has the highest silhouette score ({scores[best_k]:.4f}).",
        "profiles": profiles,
        "interpretation": "Clusters group farms by growing conditions only (targets were not used). "
                          "post-hoc yield is shown for description, not for fitting.",
    }
    coords = PCA(n_components=2, random_state=seed).fit_transform(Z)
    return pipe, report, pipe.predict(X_test), coords, km.labels_
