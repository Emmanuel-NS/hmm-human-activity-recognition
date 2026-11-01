"""
Import Sensor Logger session folders into flat labelled CSVs.

Source layout (Drive download):
  jumpig position/<session>/Accelerometer.csv + Gyroscope.csv
  Standing position/...
  still location/...
  Walking/...

Output:
  data/raw/<activity>/*.csv
  data/test/<activity>/*.csv   (last N valid sessions held out)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(r"C:\Users\PC\Downloads\drive-download-20261101T083528Z-1-001")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TEST = ROOT / "data" / "test"

FOLDER_MAP = {
    "jumpig position": "jumping",
    "Standing position": "standing",
    "still location": "still",
    "Walking": "walking",
}

MIN_DURATION_S = 4.5
TEST_PER_ACTIVITY = 2
TARGET_TRAIN_HINT = 13  # keep plenty; rubric wants ~50 total well-labelled files


def _read_meta(session: Path) -> dict:
    meta_path = session / "Metadata.csv"
    info = {"device": "unknown", "sampleRateMs": None, "platform": "unknown"}
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return info
    try:
        df = pd.read_csv(meta_path)
        if len(df):
            row = df.iloc[0]
            info["device"] = str(row.get("device name", "unknown"))
            info["platform"] = str(row.get("platform", "unknown"))
            # sampleRateMs may be pipe-separated per sensor
            srm = str(row.get("sampleRateMs", ""))
            nums = [float(x) for x in srm.split("|") if x.strip().replace(".", "", 1).isdigit()]
            if nums:
                info["sampleRateMs"] = float(np.median(nums))
    except Exception:
        pass
    return info


def merge_session(session: Path) -> pd.DataFrame | None:
    acc_path = session / "Accelerometer.csv"
    gyro_path = session / "Gyroscope.csv"
    if not acc_path.exists() or acc_path.stat().st_size < 50:
        return None
    if not gyro_path.exists() or gyro_path.stat().st_size < 50:
        return None

    try:
        acc = pd.read_csv(acc_path)
        gyro = pd.read_csv(gyro_path)
    except Exception:
        return None

    need_acc = {"seconds_elapsed", "x", "y", "z"}
    need_gyro = {"seconds_elapsed", "x", "y", "z"}
    if not need_acc.issubset(acc.columns) or not need_gyro.issubset(gyro.columns):
        return None
    if len(acc) < 30 or len(gyro) < 30:
        return None

    acc = acc[["seconds_elapsed", "x", "y", "z"]].dropna().sort_values("seconds_elapsed")
    gyro = gyro[["seconds_elapsed", "x", "y", "z"]].dropna().sort_values("seconds_elapsed")

    # interpolate gyro onto accelerometer timestamps
    t = acc["seconds_elapsed"].to_numpy(dtype=float)
    duration = float(t[-1] - t[0])
    if duration < MIN_DURATION_S:
        return None

    tg = gyro["seconds_elapsed"].to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "seconds_elapsed": t - t[0],
            "accelerometer_x": acc["x"].to_numpy(dtype=float),
            "accelerometer_y": acc["y"].to_numpy(dtype=float),
            "accelerometer_z": acc["z"].to_numpy(dtype=float),
            "gyroscope_x": np.interp(t, tg, gyro["x"].to_numpy(dtype=float)),
            "gyroscope_y": np.interp(t, tg, gyro["y"].to_numpy(dtype=float)),
            "gyroscope_z": np.interp(t, tg, gyro["z"].to_numpy(dtype=float)),
        }
    )
    # drop startup / trailing edge if needed — keep full usable span
    return out


def clear_activity_dirs() -> None:
    for split in (RAW, TEST):
        for activity in FOLDER_MAP.values():
            d = split / activity
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    clear_activity_dirs()
    summary = []
    member_rates = {}

    for src_folder, activity in FOLDER_MAP.items():
        src = SOURCE / src_folder
        sessions = sorted([p for p in src.iterdir() if p.is_dir()])
        merged = []
        for session in sessions:
            df = merge_session(session)
            if df is None:
                continue
            meta = _read_meta(session)
            # native fs from data
            t = df["seconds_elapsed"].to_numpy()
            dt = np.median(np.diff(t))
            fs = float(1.0 / dt) if dt > 0 else float("nan")
            if meta.get("device"):
                member_rates[meta["device"]] = {
                    "sampleRateMs": meta.get("sampleRateMs"),
                    "approx_fs_hz": round(fs, 1),
                    "platform": meta.get("platform"),
                }
            merged.append(
                {
                    "session": session.name,
                    "df": df,
                    "fs": fs,
                    "duration": float(t[-1] - t[0]),
                    "device": meta.get("device"),
                }
            )

        # hold out last TEST_PER_ACTIVITY for unseen eval
        if len(merged) <= TEST_PER_ACTIVITY + 2:
            raise RuntimeError(f"Not enough valid {activity} sessions: {len(merged)}")

        test_items = merged[-TEST_PER_ACTIVITY:]
        train_items = merged[:-TEST_PER_ACTIVITY]

        # Cap extremely large sets for tidy repo while staying well above rubric (≥50 files,
        # ≥90s/activity). Keep first 20 train takes when available.
        if len(train_items) > 20:
            train_items = train_items[:20]

        for i, item in enumerate(train_items, start=1):
            fs_tag = int(round(item["fs"])) if np.isfinite(item["fs"]) else 0
            path = RAW / activity / f"{activity}_{i:02d}_{item['session']}_fs{fs_tag}.csv"
            item["df"].to_csv(path, index=False)

        for i, item in enumerate(test_items, start=1):
            fs_tag = int(round(item["fs"])) if np.isfinite(item["fs"]) else 0
            path = TEST / activity / f"{activity}_test_{i:02d}_{item['session']}_fs{fs_tag}.csv"
            item["df"].to_csv(path, index=False)

        total_train_s = sum(x["duration"] for x in train_items)
        total_test_s = sum(x["duration"] for x in test_items)
        summary.append(
            {
                "activity": activity,
                "valid_sessions_found": len(merged),
                "train_files": len(train_items),
                "test_files": len(test_items),
                "train_seconds": round(total_train_s, 1),
                "test_seconds": round(total_test_s, 1),
                "mean_fs_hz": round(float(np.mean([x["fs"] for x in train_items + test_items])), 1),
            }
        )
        print(
            f"{activity}: train={len(train_items)} ({total_train_s:.0f}s), "
            f"test={len(test_items)} ({total_test_s:.0f}s), "
            f"valid={len(merged)}"
        )

    out_summary = {
        "source": str(SOURCE),
        "devices": member_rates,
        "activities": summary,
        "notes": (
            "Accelerometer + Gyroscope merged on accelerometer timestamps; "
            "sessions shorter than "
            f"{MIN_DURATION_S}s or with empty CSVs skipped."
        ),
    }
    (ROOT / "report" / "data_import_summary.json").write_text(
        json.dumps(out_summary, indent=2), encoding="utf-8"
    )
    print("Wrote", ROOT / "report" / "data_import_summary.json")
    print("Devices:", json.dumps(member_rates, indent=2))


if __name__ == "__main__":
    main()
