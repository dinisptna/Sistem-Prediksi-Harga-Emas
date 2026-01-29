import numpy as np
import pandas as pd
import joblib

# fitur minimal yang kita bisa buat dari yfinance + rolling
FEATURES = [
    "Close", "Volume",
    "daily_return", "price_ma5_ratio", "price_ma20_ratio",
    "volatility_10",
    "momentum_5_pct",
]

def compute_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()

    df["daily_return"] = df["Close"].pct_change()

    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    df["price_ma5_ratio"] = df["Close"] / df["ma_5"]
    df["price_ma20_ratio"] = df["Close"] / df["ma_20"]

    df["volatility_10"] = df["daily_return"].rolling(10).std()
    df["momentum_5_pct"] = df["Close"].pct_change(5)

    # target klasifikasi (1=naik, 0=turun/tetap)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # diperlukan untuk median return (buat proyeksi harga dari arah)
    df["return_tomorrow"] = df["Close"].shift(-1) / df["Close"] - 1

    df = df.dropna().copy()
    return df

def load_model(path: str):
    obj = joblib.load(path)
    return obj

def get_median_returns(feature_df: pd.DataFrame) -> dict:
    if "target" not in feature_df.columns or "return_tomorrow" not in feature_df.columns:
        return {"median_up": 0.0, "median_down": 0.0}
    up = feature_df.loc[feature_df["target"] == 1, "return_tomorrow"]
    dn = feature_df.loc[feature_df["target"] == 0, "return_tomorrow"]
    med_up = float(np.median(up)) if len(up) else 0.0
    med_dn = float(np.median(dn)) if len(dn) else 0.0
    return {"median_up": med_up, "median_down": med_dn}

def _get_expected_feature_names(model) -> list[str] | None:
    """
    Try to detect the feature names the model expects.
    Works for sklearn estimators (feature_names_in_) and some Pipelines.
    """
    # sklearn estimator
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Pipeline: try last step
    if hasattr(model, "named_steps"):
        for _, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def build_X_latest(latest_row: pd.Series, expected: list[str] | None) -> pd.DataFrame:
    """
    Build 1-row DataFrame with columns aligned to expected feature list.
    Missing columns will be filled with 0.0.
    """
    latest_dict = latest_row.to_dict()

    if expected is None:
        # fallback: use FEATURES yang tersedia
        cols = [c for c in FEATURES if c in latest_row.index]
        X = pd.DataFrame([{c: float(latest_dict.get(c, 0.0)) for c in cols}])
        return X

    # align to expected
    row = {}
    for c in expected:
        v = latest_dict.get(c, 0.0)
        # pastikan numeric
        try:
            row[c] = float(v)
        except Exception:
            row[c] = 0.0
    X = pd.DataFrame([row], columns=expected)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

def predict_direction(model, latest_features_row: pd.Series) -> int:
    """
    Support:
    - sklearn model langsung
    - dict bundle: {"model": estimator, ...}
    """

    # 🔹 kalau model berupa dict (bundle)
    if isinstance(model, dict):
        if "model" in model:
            estimator = model["model"]
        else:
            raise ValueError("Model dict tidak memiliki key 'model'")
    else:
        estimator = model

    expected = _get_expected_feature_names(estimator)
    X_latest = build_X_latest(latest_features_row, expected)

    pred = estimator.predict(X_latest)[0]
    return int(pred)


def direction_to_return(pred_dir: int, med: dict) -> float:
    return float(med["median_up"] if pred_dir == 1 else med["median_down"])

def predict_close_H_and_H1(last_close: float, pred_dir: int, med: dict) -> tuple[float, float, float]:
    r = direction_to_return(pred_dir, med)
    pred_H = float(last_close * (1 + r))
    pred_H1 = float(pred_H * (1 + r))  # Opsi 1: masih basis H-1
    return pred_H, pred_H1, r