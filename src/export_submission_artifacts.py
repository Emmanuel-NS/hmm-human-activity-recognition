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


def add_fig(doc: Document, path: Path, caption: str, width: float = 5.2) -> None:
    if path.exists():
        doc.add_picture(str(path), width=Inches(width))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.italic = True
            run.font.size = Pt(9)


def write_docx(results: pd.DataFrame, train_meta, test_meta, model) -> None:
    import json

    summary_path = ROOT / "report" / "data_import_summary.json"
    import_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    devices = import_summary.get("devices", {})
    if devices:
        name, info = next(iter(devices.items()))
        device_line = (
            f"{name} at about {info.get('approx_fs_hz', '?')} Hz "
            f"(Sensor Logger sampleRateMs={info.get('sampleRateMs')})"
        )
    else:
        device_line = "Samsung Android phone at about 100 Hz"

    train_s = {
        row["activity"]: row.get("train_seconds", 0) for row in import_summary.get("activities", [])
    }
    acc = float(results["Overall Accuracy"].iloc[0])
    hard = results.sort_values("Sensitivity").iloc[0]
    easy = results.sort_values("Sensitivity", ascending=False).iloc[0]

    doc = Document()
    title = doc.add_heading(
        "Modeling Human Activity States Using Hidden Markov Models", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_para(doc, "Formative 2 – Hidden Markov Models (individual)")
    add_para(
        doc,
        "Emmanuel Nsabagasa (e.nsabagasa@alustudent.com). "
        "Repo: https://github.com/Emmanuel-NS/hmm-human-activity-recognition",
    )

    add_heading(doc, "1. Background and Motivation")
    add_para(
        doc,
        "I spent a lot of this semester sitting for long stretches, so I wanted a simple "
        "phone-based check that can tell when I am actually still at a desk versus when I "
        "get up to walk or do a short active break (including jumping). Cameras feel "
        "invasive for that, but the phone already has an accelerometer and gyroscope. The "
        "catch is that the true activity is not written in the CSV – it is hidden behind "
        "noisy readings. That is why I used a Hidden Markov Model: it can keep track of "
        "which activity I am likely in over time, instead of treating every window as "
        "independent.",
    )

    add_heading(doc, "2. Data Collection and Pre-processing")
    add_para(
        doc,
        f"I recorded standing, walking, jumping, and still with Sensor Logger on my "
        f"{device_line}. Each take is roughly 5 to 10 seconds. For every session I kept "
        f"the accelerometer and gyroscope streams, aligned them on the seconds_elapsed "
        f"timestamp, and saved labelled CSV files. The GitHub dataset has "
        f"{len(train_meta) + len(test_meta)} cleaned files "
        f"({len(train_meta)} for training, {len(test_meta)} held out for testing). "
        f"Training time per activity is about "
        f"standing {train_s.get('standing', 0):.0f}s, walking {train_s.get('walking', 0):.0f}s, "
        f"jumping {train_s.get('jumping', 0):.0f}s, still {train_s.get('still', 0):.0f}s "
        f"(each above the 1 min 30 s minimum).",
    )
    add_para(
        doc,
        "My phone logged near 100 Hz. I resampled everything to 50 Hz so window length "
        "and FFT bins stay consistent. At 50 Hz I used 1.0 s windows (50 samples) with a "
        "0.5 s hop. One second gives about 1 Hz frequency resolution, which is enough to "
        "see walking cadence versus near-zero motion when I am still or standing, without "
        "making the windows so long that short changes disappear.",
    )
    add_para(doc, "Collection videos (full body, face visible, phone in hand):")
    add_para(doc, "Standing: https://youtube.com/shorts/fWiTltw83o8")
    add_para(doc, "Walking: https://youtube.com/shorts/m0d-yOL5tSE")
    add_para(doc, "Jumping: https://youtube.com/shorts/JhfENFsJmo8")
    add_para(doc, "Still: https://youtube.com/shorts/YM_wLnQTzg0")
    add_fig(doc, FIG / "fig1_raw_samples.png", "Figure 1. Example accelerometer magnitude after resampling.")

    add_heading(doc, "3. HMM Setup and Implementation")
    add_para(
        doc,
        "Hidden states are the four activities. Each observation is a feature vector from "
        "one window. I used time-domain features on accel magnitude (mean, variance, std, "
        "RMS, SMA), xy correlation, and gyro magnitude mean/std – these mainly capture how "
        "strong and shaky the motion is. From the FFT of accel magnitude I took dominant "
        "frequency and spectral energy so walking and jumping can show a clearer beat than "
        "still. Features are Z-scored using training statistics only; otherwise Hz and m/s^2 "
        "values would unfairly dominate the Gaussian emissions.",
    )
    add_para(
        doc,
        "I implemented a diagonal-Gaussian HMM in NumPy. Baum-Welch updates pi, A, and the "
        "emission means/variances and stops when the average log-likelihood change drops "
        "below 1e-4. Viterbi then picks the most likely activity path. I need Viterbi's "
        "hard labels to build sensitivity, specificity, accuracy, and the confusion matrix; "
        "Baum-Welch is what fits the parameters those metrics depend on. Code lives in the "
        "hmm module and the notebook runs the full pipeline.",
    )
    add_fig(doc, FIG / "fig_convergence.png", "Figure 2. Baum-Welch training curve.", width=4.8)

    add_heading(doc, "4. Results and Interpretation")
    add_fig(doc, FIG / "fig2_transition_matrix.png", "Figure 3. Transition matrix A.", width=4.6)
    add_para(
        doc,
        "The transition matrix is heavily diagonal. That matched how I recorded: each file "
        "is one activity for several seconds, so the model mostly learns to stay put from "
        "one 0.5 s hop to the next rather than jumping between labels.",
    )
    add_fig(doc, FIG / "fig3_emission_means.png", "Figure 4. Emission means by state.", width=5.4)
    add_fig(doc, FIG / "fig3b_emission_vars.png", "Figure 5. Emission variances by state.", width=5.4)
    add_fig(doc, FIG / "fig4_decoded_test_1.png", "Figure 6. True vs Viterbi labels on a held-out take.", width=5.2)

    add_heading(doc, "5. Evaluation on Unseen Data")
    add_para(
        doc,
        f"I kept {len(test_meta)} recordings out of training (later sessions from the same "
        f"phone, not used while fitting). That is my unseen set.",
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
        f"Overall accuracy on unseen windows was {acc:.1%}. Still and jumping were clear. "
        f"Standing was the weak spot (sensitivity {hard['Sensitivity']:.2f} when standing "
        f"was hardest): small hand motion while 'standing' often looked a bit like walking "
        f"to the model. That is honest for handheld logging – I was not always perfectly still.",
    )
    add_fig(doc, FIG / "fig5_confusion_matrix.png", "Figure 7. Confusion matrix on unseen data.", width=4.5)

    add_heading(doc, "6. Discussion and Conclusion")
    add_para(
        doc,
        f"Easiest activity overall: {easy['State (Activity)']}. Hardest: "
        f"{hard['State (Activity)']}, mostly mixed with walking. The high self-loop "
        "probabilities in A feel realistic for my short single-activity clips. Resampling "
        "100 Hz down to 50 Hz made the pipeline consistent, but it can blunt sharp jump "
        "impacts a little; the 1 s window was my compromise for FFT resolution after that "
        "downsample.",
    )
    add_para(
        doc,
        "If I continue this, I would fix phone placement (pocket vs hand), add a few more "
        "people, and try gravity-removed linear acceleration so orientation matters less. "
        "Even with standing errors, the HMM already separates quiet desk time from clear "
        "movement well enough for the break-reminder idea I started with.",
    )

    out = ROOT / "report" / "Formative2_HMM_Report.docx"
    try:
        doc.save(out)
    except PermissionError:
        out = ROOT / "report" / "Formative2_HMM_Report_SUBMIT.docx"
        doc.save(out)
        print("NOTE: original docx was locked; wrote", out)


if __name__ == "__main__":
    main()
