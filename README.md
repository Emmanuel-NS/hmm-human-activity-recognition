# hmm-human-activity-recognition

Recognise four phone activities — **standing**, **walking**, **jumping**, **still** — from accelerometer and gyroscope CSVs using a Gaussian Hidden Markov Model written in NumPy (Baum-Welch + Viterbi).

Data was recorded with Sensor Logger on a Samsung SM-A325N (~100 Hz), then resampled to 50 Hz. Features are taken from 1 s windows (0.5 s hop). On held-out files overall accuracy was about 83%.

## Repo structure

```
data/raw/<activity>/*.csv          training recordings
data/test/<activity>/*.csv         unseen test recordings
notebooks/hmm_activity_recognition.ipynb
src/
  hmm.py                           Baum-Welch + Viterbi
  features.py                      time + FFT features, Z-score
  data_utils.py                    load, resample, window
  metrics.py                       sensitivity / specificity / accuracy
report/Formative2_HMM_Report.docx  written report
requirements.txt
```

## Requirements

- Python 3.10+
- packages in `requirements.txt` (numpy, pandas, matplotlib, seaborn, scipy, jupyter, …)

## How to run

```bash
# 1. install deps
pip install -r requirements.txt

# 2. open the notebook
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

Run all cells top to bottom. The notebook will:

1. load CSVs from `data/raw` and `data/test`
2. resample to 50 Hz and build window features
3. train the HMM with Baum-Welch
4. decode with Viterbi
5. print the metrics table and show the plots

If you open the notebook from another working directory, it still finds the project root as long as the `notebooks/` and `data/` folders sit next to each other as in this repo.

## Using your own recordings

1. Export Sensor Logger (or similar) CSVs with a time column plus accel xyz and gyro xyz.
2. Put training files in `data/raw/<standing|walking|jumping|still>/`.
3. Put held-out files in `data/test/<same activity names>/`.
4. Re-run the notebook.

Column names like `seconds_elapsed`, `accelerometer_x`, `gyroscope_x` are handled automatically; other common headers usually work too via the loader in `src/data_utils.py`.

## Notes

- Native rate here was ~100 Hz (`sampleRateMs=10`); everything is resampled to **50 Hz** before windowing.
- Window = **1.0 s**, hop = **0.5 s**.
- Report file for submission: `report/Formative2_HMM_Report.docx`.
