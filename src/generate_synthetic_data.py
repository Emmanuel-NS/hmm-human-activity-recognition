"""
Generate Sensor-Logger-like CSV files so the pipeline runs before real phone data.

Replace these with your real recordings under data/raw/<activity>/ and
data/test/<activity>/. Keep the same column names if possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TEST = ROOT / "data" / "test"

ACTIVITIES = ["standing", "walking", "jumping", "still"]

# Per-activity generative parameters (accel in m/s^2, gyro in rad/s)
PARAMS = {
    "still": dict(acc_mean=(0, 0, 9.81), acc_noise=0.02, gyro_noise=0.01, wobble=0.0, hop=0.0),
    "standing": dict(acc_mean=(0.1, 0.05, 9.81), acc_noise=0.08, gyro_noise=0.03, wobble=0.15, hop=0.0),
    "walking": dict(acc_mean=(0.2, 0.1, 9.81), acc_noise=0.25, gyro_noise=0.2, wobble=1.5, hop=1.8),
    "jumping": dict(acc_mean=(0.3, 0.2, 9.81), acc_noise=0.5, gyro_noise=0.45, wobble=4.0, hop=2.5),
}


def synthesize(activity: str, duration_s: float, fs: float, rng: np.random.Generator) -> pd.DataFrame:
    p = PARAMS[activity]
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    ax = np.full(n, p["acc_mean"][0], dtype=float)
    ay = np.full(n, p["acc_mean"][1], dtype=float)
    az = np.full(n, p["acc_mean"][2], dtype=float)
    gx = np.zeros(n)
    gy = np.zeros(n)
    gz = np.zeros(n)

    if p["hop"] > 0:
        ax += p["wobble"] * np.sin(2 * np.pi * p["hop"] * t)
        ay += 0.6 * p["wobble"] * np.sin(2 * np.pi * p["hop"] * t + 0.4)
        az += p["wobble"] * np.sin(2 * np.pi * p["hop"] * t + 1.1)
        gx += 0.3 * p["wobble"] * np.sin(2 * np.pi * p["hop"] * t)
        gy += 0.25 * p["wobble"] * np.cos(2 * np.pi * p["hop"] * t)
        gz += 0.2 * p["wobble"] * np.sin(2 * np.pi * (p["hop"] * 0.5) * t)
    elif p["wobble"] > 0:
        # standing micro-movements
        ax += p["wobble"] * np.sin(2 * np.pi * 0.4 * t)
        ay += 0.5 * p["wobble"] * np.cos(2 * np.pi * 0.35 * t)

    ax += rng.normal(0, p["acc_noise"], n)
    ay += rng.normal(0, p["acc_noise"], n)
    az += rng.normal(0, p["acc_noise"], n)
    gx += rng.normal(0, p["gyro_noise"], n)
    gy += rng.normal(0, p["gyro_noise"], n)
    gz += rng.normal(0, p["gyro_noise"], n)

    return pd.DataFrame(
        {
            "timestamp": t,
            "acc_x": ax,
            "acc_y": ay,
            "acc_z": az,
            "gyro_x": gx,
            "gyro_y": gy,
            "gyro_z": gz,
        }
    )


def main() -> None:
    rng = np.random.default_rng(7)

    # Rubric target: ~50 well-labelled files, >= ~90s per activity overall
    # 12 train files × ~8s ≈ 96s per activity; 2 test files each
    train_per_activity = 12
    test_per_activity = 2
    duration = 8.0
    # Slightly different native rates to practice harmonization
    member_rates = [50.0, 50.0, 100.0, 50.0]

    for split_root, n_files in ((RAW, train_per_activity), (TEST, test_per_activity)):
        for activity in ACTIVITIES:
            out_dir = split_root / activity
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_files):
                fs = member_rates[i % len(member_rates)]
                df = synthesize(activity, duration, fs, rng)
                # Sensor Logger-ish column aliases in header
                df = df.rename(
                    columns={
                        "timestamp": "seconds_elapsed",
                        "acc_x": "accelerometer_x",
                        "acc_y": "accelerometer_y",
                        "acc_z": "accelerometer_z",
                        "gyro_x": "gyroscope_x",
                        "gyro_y": "gyroscope_y",
                        "gyro_z": "gyroscope_z",
                    }
                )
                path = out_dir / f"{activity}_{i+1:02d}_fs{int(fs)}.csv"
                df.to_csv(path, index=False)

    print(f"Wrote synthetic train CSVs under {RAW}")
    print(f"Wrote synthetic test CSVs under {TEST}")
    print("Replace with real Sensor Logger exports before final submission.")


if __name__ == "__main__":
    main()
