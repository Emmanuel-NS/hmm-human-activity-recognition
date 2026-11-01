"""Load Sensor Logger / Physics Toolbox CSV files and build windowed sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, extract_window_features, zscore_normalize

ACTIVITY_TO_ID = {
    "standing": 0,
    "walking": 1,
    "jumping": 2,
    "still": 3,
}
ID_TO_ACTIVITY = {v: k for k, v in ACTIVITY_TO_ID.items()}


def _guess_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map flexible Sensor Logger column names to standard keys."""
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    def find(*candidates: str) -> str | None:
        for cand in candidates:
            for key, orig in cols.items():
                if cand in key.replace(" ", "").replace("_", ""):
                    return orig
        return None

    # timestamp
    for key, orig in cols.items():
        if "time" in key or key in ("t", "seconds_elapsed"):
            mapping["timestamp"] = orig
            break

    mapping["ax"] = find("accx", "accelerometerx", "ax", "x-axis") or find("x")
    mapping["ay"] = find("accy", "accelerometery", "ay", "y-axis") or None
    mapping["az"] = find("accz", "accelerometerz", "az", "z-axis") or None
    mapping["gx"] = find("gyrox", "gyroscopex", "gx") or None
    mapping["gy"] = find("gyroy", "gyroscopey", "gy") or None
    mapping["gz"] = find("gyroz", "gyroscopez", "gz") or None

    # Fallback for Physics Toolbox / simple 3-axis accel files
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if mapping.get("ax") is None and len(numeric) >= 3:
        # assume last 3 numeric are xyz or first timestamp then xyz
        if "timestamp" in mapping and mapping["timestamp"] in numeric:
            numeric = [c for c in numeric if c != mapping["timestamp"]]
        mapping["ax"], mapping["ay"], mapping["az"] = numeric[0], numeric[1], numeric[2]
        if len(numeric) >= 6:
            mapping["gx"], mapping["gy"], mapping["gz"] = numeric[3], numeric[4], numeric[5]

    return mapping


def load_recording(path: str | Path) -> pd.DataFrame:
    """Load one CSV/TXT recording and return standardized columns."""
    path = Path(path)
    df = pd.read_csv(path)
    mapping = _guess_columns(df)

    out = pd.DataFrame()
    if "timestamp" in mapping:
        out["timestamp"] = pd.to_numeric(df[mapping["timestamp"]], errors="coerce")
    else:
        out["timestamp"] = np.arange(len(df), dtype=float)

    for src, dst in [
        ("ax", "acc_x"),
        ("ay", "acc_y"),
        ("az", "acc_z"),
        ("gx", "gyro_x"),
        ("gy", "gyro_y"),
        ("gz", "gyro_z"),
    ]:
        col = mapping.get(src)
        if col is not None:
            out[dst] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[dst] = 0.0

    out = out.dropna().reset_index(drop=True)
    return out


def estimate_sampling_rate(timestamps: np.ndarray) -> float:
    """Estimate Hz from timestamp column (seconds or milliseconds)."""
    ts = np.asarray(timestamps, dtype=float)
    dts = np.diff(ts)
    dts = dts[dts > 0]
    if len(dts) == 0:
        return 50.0
    median_dt = float(np.median(dts))
    # if timestamps look like milliseconds
    if median_dt > 1.0:
        median_dt = median_dt / 1000.0
    return 1.0 / median_dt


def resample_dataframe(df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
    """
    Harmonize sampling rate by linear interpolation onto a uniform grid.
    Use the same target_fs for every phone/member.
    """
    ts = df["timestamp"].to_numpy(dtype=float)
    # normalize to seconds starting at 0
    if np.median(np.diff(ts)) > 1.0:
        ts = ts / 1000.0
    ts = ts - ts[0]
    duration = ts[-1]
    if duration <= 0:
        return df.copy()

    new_t = np.arange(0.0, duration, 1.0 / target_fs)
    out = {"timestamp": new_t}
    for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        out[col] = np.interp(new_t, ts, df[col].to_numpy(dtype=float))
    return pd.DataFrame(out)


def window_indices(n_samples: int, window_size: int, hop: int) -> list[tuple[int, int]]:
    starts = range(0, max(n_samples - window_size + 1, 1), hop)
    return [(s, s + window_size) for s in starts if s + window_size <= n_samples]


def file_to_feature_sequence(
    path: str | Path,
    activity: str,
    target_fs: float = 50.0,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert one recording into a sequence of feature vectors + labels.

    Window size logic (document in report):
      window_samples = window_sec * target_fs
    At 50 Hz, 1.0 s → 50 samples: enough for FFT frequency resolution (~1 Hz)
    while short enough to capture transitions between activities.
    """
    df = load_recording(path)
    native_fs = estimate_sampling_rate(df["timestamp"].to_numpy())
    df = resample_dataframe(df, target_fs)

    window_size = max(int(round(window_sec * target_fs)), 8)
    hop = max(int(round(hop_sec * target_fs)), 1)

    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()

    feats = []
    for start, end in window_indices(len(df), window_size, hop):
        feats.append(extract_window_features(acc[start:end], gyro[start:end], target_fs))

    if not feats:
        # file too short — one feature vector from whole file
        feats.append(extract_window_features(acc, gyro, target_fs))

    X = np.vstack(feats)
    y = np.full(len(X), ACTIVITY_TO_ID[activity], dtype=int)
    return X, y, native_fs


def load_dataset(
    root: str | Path,
    target_fs: float = 50.0,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """
    Load all CSVs under root/<activity>/*.csv

    Returns lists of per-file feature sequences, label sequences, and metadata.
    """
    root = Path(root)
    sequences_X: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    meta: list[dict] = []

    for activity in ACTIVITY_TO_ID:
        folder = root / activity
        if not folder.exists():
            continue
        files = sorted(list(folder.glob("*.csv")) + list(folder.glob("*.txt")))
        for f in files:
            X, y, native_fs = file_to_feature_sequence(
                f, activity, target_fs=target_fs, window_sec=window_sec, hop_sec=hop_sec
            )
            sequences_X.append(X)
            sequences_y.append(y)
            meta.append(
                {
                    "file": str(f),
                    "activity": activity,
                    "native_fs_hz": native_fs,
                    "n_windows": len(X),
                }
            )
    return sequences_X, sequences_y, meta


def prepare_xy(
    sequences_X: list[np.ndarray],
    sequences_y: list[np.ndarray],
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Stack + Z-score; also return normalized per-file sequences."""
    X_all = np.vstack(sequences_X)
    y_all = np.concatenate(sequences_y)
    X_norm, mean, std = zscore_normalize(X_all, mean, std)

    # rebuild per-file normalized sequences
    norm_seqs = []
    offset = 0
    for seq in sequences_X:
        n = len(seq)
        norm_seqs.append(X_norm[offset : offset + n])
        offset += n

    return X_norm, y_all, mean, std, norm_seqs, sequences_y


__all__ = [
    "ACTIVITY_TO_ID",
    "ID_TO_ACTIVITY",
    "FEATURE_NAMES",
    "load_recording",
    "estimate_sampling_rate",
    "resample_dataframe",
    "file_to_feature_sequence",
    "load_dataset",
    "prepare_xy",
]
