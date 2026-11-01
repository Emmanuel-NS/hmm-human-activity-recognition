"""Train HMM, save figures + metrics, and write the Word report for LMS upload."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import (  # noqa: E402
    ID_TO_ACTIVITY,
    estimate_sampling_rate,
    load_dataset,
    load_recording,
    prepare_xy,
    resample_dataframe,
)
from features import FEATURE_NAMES  # noqa: E402
from hmm import GaussianHMM  # noqa: E402
from metrics import confusion_matrix, per_class_sensitivity_specificity  # noqa: E402

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
STATE_NAMES = [ID_TO_ACTIVITY[i] for i in range(4)]
TARGET_FS = 50.0


def save_raw_plots() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, activity in zip(axes.ravel(), STATE_NAMES):
        files = sorted((ROOT / "data" / "raw" / activity).glob("*.csv"))
        df = load_recording(files[0])
        fs = estimate_sampling_rate(df["timestamp"].to_numpy())
        df = resample_dataframe(df, TARGET_FS)
        mag = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
        ax.plot(df["timestamp"], mag, lw=0.9)
        ax.set_title(f"{activity} (native ~{fs:.0f} Hz -> {TARGET_FS:.0f} Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("|a|")
    fig.suptitle("Sample accelerometer magnitude per activity")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_raw_samples.png", dpi=160)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    train_X, train_y, train_meta = load_dataset(
        ROOT / "data" / "raw", target_fs=TARGET_FS, window_sec=1.0, hop_sec=0.5
    )
    test_X, test_y, test_meta = load_dataset(
        ROOT / "data" / "test", target_fs=TARGET_FS, window_sec=1.0, hop_sec=0.5
    )
    Xtr, ytr, mean, std, tr_seqs, tr_labs = prepare_xy(train_X, train_y)
    _, _, _, _, te_seqs, te_labs = prepare_xy(test_X, test_y, mean=mean, std=std)

    save_raw_plots()

    model = GaussianHMM(4, Xtr.shape[1], STATE_NAMES, 42)
    model.initialize(Xtr, ytr)
    model.fit(tr_seqs, max_iter=80, tol=1e-4, verbose=True)

    # Convergence
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(model.log_likelihoods_, marker="o", ms=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg log-likelihood")
    ax.set_title("Baum-Welch convergence")
    fig.tight_layout()
    fig.savefig(FIG / "fig_convergence.png", dpi=160)
    plt.close(fig)

    # Transition heatmap
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        model.A,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=STATE_NAMES,
        yticklabels=STATE_NAMES,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Transition matrix A")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_transition_matrix.png", dpi=160)
    plt.close(fig)

    # Emission means
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(
        model.means,
        cmap="coolwarm",
        center=0,
        yticklabels=STATE_NAMES,
        xticklabels=FEATURE_NAMES,
        ax=ax,
    )
    ax.set_title("Emission means (Z-scored features)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_emission_means.png", dpi=160)
    plt.close(fig)

    # Emission vars
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(
        model.vars,
        cmap="YlOrRd",
        yticklabels=STATE_NAMES,
        xticklabels=FEATURE_NAMES,
        ax=ax,
    )
    ax.set_title("Emission variances")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3b_emission_vars.png", dpi=160)
    plt.close(fig)

    # Decoded test sequences (first 4)
    for i, (seq_X, seq_y) in enumerate(zip(te_seqs[:4], te_labs[:4])):
        pred, logp = model.decode(seq_X)
        t = np.arange(len(pred))
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.step(t, seq_y, where="mid", label="True", lw=2)
        ax.step(t, pred, where="mid", label="Viterbi", lw=2, alpha=0.85)
        ax.set_yticks(range(4))
        ax.set_yticklabels(STATE_NAMES)
        act = STATE_NAMES[int(seq_y[0])]
        ax.set_title(f"Decoded unseen sequence ({act}) | logp={logp:.1f}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG / f"fig4_decoded_test_{i+1}.png", dpi=160)
        plt.close(fig)

    y_true = np.concatenate(te_labs)
    y_pred = np.concatenate([model.predict(s) for s in te_seqs])
    cm = confusion_matrix(y_true, y_pred, 4)
    results = per_class_sensitivity_specificity(y_true, y_pred, 4, STATE_NAMES)
    results.to_csv(ROOT / "report" / "evaluation_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=STATE_NAMES,
        yticklabels=STATE_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (unseen test)")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_confusion_matrix.png", dpi=160)
    plt.close(fig)

    write_docx(results, train_meta, test_meta, model)
    print("Wrote figures to", FIG)
    print("Wrote report to", ROOT / "report" / "Formative2_HMM_Report.docx")
    print(results.to_string(index=False))


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def add_para(doc: Document, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(11)


def add_fig(doc: Document, path: Path, caption: str, width: float = 5.8) -> None:
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.italic = True
            run.font.size = Pt(10)


def write_docx(results: pd.DataFrame, train_meta, test_meta, model) -> None:
    import json

    summary_path = ROOT / "report" / "data_import_summary.json"
    import_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    devices = import_summary.get("devices", {})
    device_line = ", ".join(
        f"{name} (~{info.get('approx_fs_hz', '?')} Hz, sampleRateMs={info.get('sampleRateMs')})"
        for name, info in devices.items()
    ) or "see Metadata.csv in Sensor Logger exports"

    doc = Document()
    title = doc.add_heading(
        "Modeling Human Activity States Using Hidden Markov Models", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_para(doc, "Formative 2 — Hidden Markov Models (Individual submission)")
    add_para(
        doc,
        "Student: Emmanuel Nsabagasa (e.nsabagasa@alustudent.com)    |    "
        "GitHub: https://github.com/Emmanuel-NS/hmm-human-activity-recognition",
        bold=True,
    )

    add_heading(doc, "1. Background and Motivation")
    add_para(
        doc,
        "Smartphone inertial sensors continuously emit accelerometer and gyroscope "
        "streams, but the true activity (standing, walking, jumping, or still) is "
        "hidden behind noise and device orientation. Hidden Markov Models (HMMs) "
        "capture both temporal persistence of activities and noisy emissions, making "
        "them suitable for sequential human activity recognition (HAR).",
    )
    add_para(
        doc,
        "Unique use case (individual): I model short indoor activity bursts for a "
        "personal student wellness / sedentary-break reminder. A phone already carried "
        "in hand or pocket should distinguish desk still/standing from walking or "
        "jumping during a movement break, without cameras. Human activity recognition "
        "is needed here so reminders fire only when the user is sedentary, and active "
        "breaks can be verified privately from IMU signals alone.",
    )

    add_heading(doc, "2. Data Collection and Pre-processing")
    act_lines = []
    for row in import_summary.get("activities", []):
        act_lines.append(
            f"{row['activity']}: {row['train_files']} train + {row['test_files']} test "
            f"({row['train_seconds']}s train, mean fs≈{row['mean_fs_hz']} Hz)"
        )
    add_para(
        doc,
        f"This is an individual submission. I collected real sequential IMU data with "
        f"Sensor Logger (Android) on one phone. Each session exports Accelerometer.csv "
        f"and Gyroscope.csv with a seconds_elapsed timestamp column; I merged gyroscope "
        f"onto accelerometer timestamps, kept takes of about 5–10 s, and wrote flat "
        f"labelled CSVs under data/raw and data/test. "
        f"In total the repository contains {len(train_meta) + len(test_meta)} well-labelled "
        f"cleaned files ({len(train_meta)} train + {len(test_meta)} unseen test), with each "
        f"activity exceeding 1 min 30 s of training signal. "
        + (" Breakdown — " + "; ".join(act_lines) + "." if act_lines else ""),
    )
    add_para(
        doc,
        "Sampling rate (individual device documented for the report): "
        f"{device_line}. "
        "Even with a single phone, Sensor Logger can be configured differently across "
        "sessions; therefore every file is linearly resampled to a common 50 Hz grid "
        "before windowing so FFT bins and feature windows stay comparable. If multiple "
        "phones/rates had been used, the same harmonisation step would align them.",
        bold=False,
    )
    add_para(
        doc,
        "Windowing: at the harmonised 50 Hz rate we use 1.0 s windows (50 samples) with "
        "0.5 s hop. Native phone rate was ~100 Hz (10 ms Sensor Logger setting); "
        "after downsampling, FFT frequency resolution is about 1 Hz — enough to separate "
        "still/standing (near 0 Hz) from walking cadence (~1.5–2 Hz) and jumping (~2–3 Hz), "
        "while remaining short enough to track transitions.",
    )
    add_para(
        doc,
        "Video evidence of data collection (4 activities; full body + face visible while "
        "holding the phone):",
        bold=True,
    )
    add_para(doc, "Standing video: https://youtube.com/shorts/fWiTltw83o8")
    add_para(doc, "Walking video: https://youtube.com/shorts/m0d-yOL5tSE")
    add_para(doc, "Jumping video: https://youtube.com/shorts/JhfENFsJmo8")
    add_para(doc, "Still video: https://youtube.com/shorts/YM_wLnQTzg0")
    add_fig(doc, FIG / "fig1_raw_samples.png", "Figure 1. Sample |a| traces after resampling.")

    add_heading(doc, "3. HMM Setup and Implementation")
    add_para(
        doc,
        "Hidden states Z = {standing, walking, jumping, still}. Observations X are 10-D "
        "Z-scored feature vectors per window. Transition matrix A stores P(z_{t+1}|z_t). "
        "Emissions B are diagonal Gaussians N(x|μ_i, σ²_i). π is the initial state distribution.",
    )
    add_para(
        doc,
        "Features — time domain: mean, variance, std, RMS, SMA of accelerometer magnitude, "
        "corr(x,y), gyro mean/std magnitude (energy and axis coupling separate locomotion "
        "from stillness). Frequency domain (FFT): dominant frequency and spectral energy "
        "(cadence / periodic power). Z-score normalisation is fit on training only so "
        "heterogeneous units do not dominate Gaussian emissions.",
    )
    add_para(
        doc,
        "Algorithms: Baum–Welch (EM) learns π, A, and Gaussian parameters with convergence "
        "when |Δll| < 1e-4. Forward–Backward supplies the E-step. Viterbi returns the MAP "
        "state path used to compute sensitivity, specificity, overall accuracy, and the "
        "confusion matrix on unseen files. Implementation is NumPy-only in src/hmm.py, "
        "executed via notebooks/hmm_activity_recognition.ipynb.",
    )
    add_fig(doc, FIG / "fig_convergence.png", "Figure 2. Baum–Welch log-likelihood convergence.")

    add_heading(doc, "4. Results and Interpretation")
    add_fig(doc, FIG / "fig2_transition_matrix.png", "Figure 3. Learned transition matrix A.")
    add_para(
        doc,
        "High diagonal mass in A reflects realistic behaviour for short continuous takes: "
        "the model prefers to remain in the current activity across consecutive 1 s windows.",
    )
    add_fig(doc, FIG / "fig3_emission_means.png", "Figure 4. Emission means (B) in Z-scored feature space.")
    add_fig(doc, FIG / "fig3b_emission_vars.png", "Figure 5. Emission variances (diagonal Gaussian B).")
    add_fig(
        doc,
        FIG / "fig4_decoded_test_1.png",
        "Figure 6. Viterbi decoding on an unseen test sequence.",
    )
    add_fig(
        doc,
        FIG / "fig4_decoded_test_2.png",
        "Figure 7. Additional decoded unseen sequence (activity transition view).",
    )

    add_heading(doc, "5. Evaluation on Unseen Data")
    add_para(
        doc,
        f"Unseen evaluation used {len(test_meta)} held-out Sensor Logger sessions in data/test "
        "(later recording folders not used during training). This is a new-session split "
        "from the same collection day/device, testing whether the HMM generalises beyond "
        "the exact takes seen in training.",
    )
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, col in enumerate(
        ["State (Activity)", "Number of Samples", "Sensitivity", "Specificity", "Overall Accuracy"]
    ):
        hdr[i].text = col
    for _, row in results.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["State (Activity)"])
        cells[1].text = str(int(row["Number of Samples"]))
        cells[2].text = f"{row['Sensitivity']:.4f}"
        cells[3].text = f"{row['Specificity']:.4f}"
        cells[4].text = f"{row['Overall Accuracy']:.4f}"

    add_para(
        doc,
        "Generalisation: still and jumping are recognised reliably on held-out takes; "
        "standing remains the main error mode (often decoded as walking when handheld "
        "micro-motion raises energy features). Overall accuracy on unseen windows is "
        f"{float(results['Overall Accuracy'].iloc[0]):.1%}, which is usable for a "
        "break-reminder filter but would benefit from placement-consistent training.",
    )
    add_fig(doc, FIG / "fig5_confusion_matrix.png", "Figure 8. Confusion matrix on unseen test data.")

    add_heading(doc, "6. Discussion and Conclusion")
    # Personalised from this run's metrics
    hard = results.sort_values("Sensitivity").iloc[0]
    easy = results.sort_values("Sensitivity", ascending=False).iloc[0]
    acc = float(results["Overall Accuracy"].iloc[0])
    add_para(
        doc,
        f"On unseen Sensor Logger takes the model reached overall accuracy {acc:.1%}. "
        f"Easiest state by sensitivity: {easy['State (Activity)']} "
        f"(sensitivity={easy['Sensitivity']:.2f}). "
        f"Hardest: {hard['State (Activity)']} (sensitivity={hard['Sensitivity']:.2f}). "
        "In our confusion matrix, standing is most often confused with walking — "
        "both can show mild phone motion and overlapping spectral energy when the user "
        "shifts weight or the phone is handheld rather than pocket-fixed. "
        "Still and jumping separate cleanly via magnitude and FFT energy.",
    )
    add_para(
        doc,
        "Transition probabilities remain strongly diagonal, matching realistic behaviour: "
        "within an 8 s labelled take the activity rarely switches every 0.5 s hop. "
        "Native sampling was ~100 Hz (SM-A325N, Sensor Logger 10 ms); resampling to 50 Hz "
        "harmonises members/devices but can soften high-frequency impact peaks in jumping. "
        "The 1.0 s window after resampling keeps ~1 Hz FFT resolution for cadence features.",
    )
    add_para(
        doc,
        "Improvements: more participants and phone placements (waist vs hand), magnetometer "
        "features, full-covariance Gaussians, orientation-invariant features (e.g., gravity-removed "
        "linear accel), and longer multi-activity sessions to enrich off-diagonal transitions. "
        "Overall, the Gaussian HMM with Baum–Welch training and Viterbi decoding recovers "
        "activity states from real IMU feature sequences and supports our wellness-break "
        "reminder use case by separating sedentary versus active phone contexts.",
    )

    out = ROOT / "report" / "Formative2_HMM_Report.docx"
    doc.save(out)


if __name__ == "__main__":
    main()
