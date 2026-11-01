"""Time-domain and frequency-domain feature extraction for HAR."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "mean_acc_mag",
    "var_acc_mag",
    "std_acc_mag",
    "sma_acc",
    "rms_acc",
    "corr_xy",
    "mean_gyro_mag",
    "std_gyro_mag",
    "dominant_freq",
    "spectral_energy",
]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_window_features(
    acc: np.ndarray,
    gyro: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Extract features from one window.

    Parameters
    ----------
    acc : (N, 3) accelerometer xyz
    gyro : (N, 3) gyroscope xyz
    fs : sampling rate in Hz

    Returns
    -------
    feature vector of shape (n_features,)
    """
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    gx, gy, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # --- Time domain ---
    mean_acc_mag = float(np.mean(acc_mag))
    var_acc_mag = float(np.var(acc_mag))
    std_acc_mag = float(np.std(acc_mag))
    # Signal Magnitude Area (normalized by window length)
    sma_acc = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    rms_acc = float(np.sqrt(np.mean(acc_mag**2)))
    corr_xy = _safe_corr(ax, ay)

    mean_gyro_mag = float(np.mean(gyro_mag))
    std_gyro_mag = float(np.std(gyro_mag))

    # --- Frequency domain (FFT on accelerometer magnitude) ---
    n = len(acc_mag)
    # remove DC for clearer activity frequency content
    signal = acc_mag - np.mean(acc_mag)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(spectrum) ** 2

    if len(power) > 1:
        # ignore DC bin when picking dominant frequency
        dominant_idx = int(np.argmax(power[1:]) + 1)
        dominant_freq = float(freqs[dominant_idx])
        spectral_energy = float(np.sum(power) / n)
    else:
        dominant_freq = 0.0
        spectral_energy = 0.0

    return np.array(
        [
            mean_acc_mag,
            var_acc_mag,
            std_acc_mag,
            sma_acc,
            rms_acc,
            corr_xy,
            mean_gyro_mag,
            std_gyro_mag,
            dominant_freq,
            spectral_energy,
        ],
        dtype=float,
    )


def zscore_normalize(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features.
    Fit on training data (pass mean/std=None), transform test with training stats.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std
