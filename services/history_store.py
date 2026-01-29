from __future__ import annotations

import uuid
from pathlib import Path
import pandas as pd


DEFAULT_PATH = Path("data/prediksi_history.csv")

COLUMNS = [
    "id",
    "tanggal_input",
    "harga_input",
    "tanggal_besok",
    "harga_besok",
    "pct_besok",
    "tanggal_minggu",
    "harga_minggu",
    "pct_minggu",
    "tanggal_bulan",
    "harga_bulan",
    "pct_bulan",
    "created_at",
    "updated_at",
]


def _ensure_parent_dir(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_history_csv(csv_path: Path = DEFAULT_PATH) -> Path:
    """Create CSV with headers if not exists."""
    _ensure_parent_dir(csv_path)
    if not csv_path.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(csv_path, index=False)
    return csv_path


def load_history(csv_path: Path = DEFAULT_PATH) -> pd.DataFrame:
    ensure_history_csv(csv_path)
    df = pd.read_csv(csv_path)

    # safe types
    if not df.empty:
        for c in ["harga_input", "harga_besok", "harga_minggu", "harga_bulan", "pct_besok", "pct_minggu", "pct_bulan"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def save_history(df: pd.DataFrame, csv_path: Path = DEFAULT_PATH) -> None:
    ensure_history_csv(csv_path)

    # pastikan kolom lengkap
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[COLUMNS].copy()
    df.to_csv(csv_path, index=False)


def add_record(record: dict, csv_path: Path = DEFAULT_PATH) -> str:
    """
    Add a new record, auto-generate id.
    Returns the id.
    """
    df = load_history(csv_path)
    new_id = uuid.uuid4().hex[:6]

    now = pd.Timestamp.now().isoformat(timespec="seconds")
    row = {c: None for c in COLUMNS}
    row.update(record)
    row["id"] = new_id
    row["created_at"] = now
    row["updated_at"] = now

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_history(df, csv_path)
    return new_id


def update_record(record_id: str, updates: dict, csv_path: Path = DEFAULT_PATH) -> bool:
    """Update by id. Returns True if updated."""
    df = load_history(csv_path)
    if df.empty or "id" not in df.columns:
        return False

    mask = df["id"].astype(str) == str(record_id)
    if not mask.any():
        return False

    for k, v in updates.items():
        if k in df.columns:
            df.loc[mask, k] = v

    df.loc[mask, "updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    save_history(df, csv_path)
    return True


def delete_record(record_id: str, csv_path: Path = DEFAULT_PATH) -> bool:
    """Delete by id. Returns True if deleted."""
    df = load_history(csv_path)
    if df.empty or "id" not in df.columns:
        return False

    before = len(df)
    df = df[df["id"].astype(str) != str(record_id)].copy()
    after = len(df)
    if after == before:
        return False

    save_history(df, csv_path)
    return True
