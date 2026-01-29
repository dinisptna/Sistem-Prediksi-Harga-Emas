import pandas as pd
import yfinance as yf

def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Kalau MultiIndex (Open/Close level 0, ticker level 1) -> buang level ticker
    if isinstance(df.columns, pd.MultiIndex):
        # Ambil level terakhir (biasanya ticker) dan drop
        # Contoh: ('Close','GC=F') -> 'Close'
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def fetch_gold_ohlcv(
    start: str = "2021-01-19",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(
        "GC=F",
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    df = _normalize_yf_columns(df)
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang dari yfinance: {missing}. Kolom yang ada: {list(df.columns)}")

    # Pastikan tipe numerik
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().copy()
    return df