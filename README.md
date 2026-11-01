# hmm-human-activity-recognition

Phone IMU activity recognition with a Gaussian HMM written in NumPy.

I logged accelerometer and gyroscope with Sensor Logger (Samsung SM-A325N, ~100 Hz) for four activities: standing, walking, jumping, and still. Signals are resampled to 50 Hz, cut into 1 s windows (0.5 s hop), turned into time and FFT features, then labelled with Baum-Welch training and Viterbi decoding.

On held-out recordings the model reached about 83% overall accuracy. Still and jumping were clear; standing was the hardest and often mixed with walking when the phone moved a little in my hand.

## Layout

| Path | Content |
|------|---------|
| `data/raw/` | 80 labelled training CSVs |
| `data/test/` | 8 unseen test CSVs |
| `notebooks/hmm_activity_recognition.ipynb` | end-to-end run |
| `src/hmm.py` | Baum-Welch + Viterbi |
| `src/features.py` | feature extraction + Z-score |
| `src/data_utils.py` | load / resample / window |
| `report/Formative2_HMM_Report.docx` | short report |

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```
