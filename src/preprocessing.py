"""One ColumnTransformer shared by training and inference (no leakage: fitted on train only)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BINARY, CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numbers, mode-impute + one-hot season, mode-impute binary."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
    ])
