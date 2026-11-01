# Formative 2 — Hidden Markov Models for Human Activity Recognition

Gaussian HMM (NumPy from scratch) that infers **standing / walking / jumping / still** from smartphone accelerometer + gyroscope CSVs.

## Repository layout

```
data/
  raw/<activity>/*.csv     # training recordings (label = folder name)
  test/<activity>/*.csv    # unseen evaluation (≥2 files)
notebooks/
  hmm_activity_recognition.ipynb
src/
  hmm.py                   # Baum–Welch + Viterbi
  features.py              # time + FFT features, Z-score
  data_utils.py            # load / resample / window
  metrics.py               # sensitivity, specificity, accuracy
  generate_synthetic_data.py
report/
  REPORT_OUTLINE.md        # paste into Word (4–5 pages)
videos/                    # 4 identity videos (not uploaded to public GitHub if private)
```

## Quick start

```bash
pip install -r requirements.txt
python src/generate_synthetic_data.py   # demo data only — replace with real Sensor Logger CSVs
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

## Real data collection checklist

1. Install **Sensor Logger** (or Physics Toolbox).
2. Record accel + gyro for each activity **5–10 s**; aim for **≥50 labelled files** and **≥90 s total per activity**.
3. Save CSV with a timestamp column.
4. Put files in `data/raw/<activity>/` and hold out ≥2 files in `data/test/<activity>/`.
5. Record **4 videos** (face visible, full body, phone in hand, good lighting) — one per activity. Link them in the report.
6. Write each member’s sampling rate in the notebook + report.
7. Push this repo to GitHub and submit the report (doc/docx) on the LMS.

## Algorithms

| Algorithm | Role |
|-----------|------|
| Baum–Welch | Train π, A, Gaussian emissions; stop when \|Δll\| < ε |
| Viterbi | Decode activity sequence → evaluation metrics |
| Forward–Backward | E-step inside Baum–Welch |
