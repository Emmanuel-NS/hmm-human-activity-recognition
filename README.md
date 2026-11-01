# Formative 2 – HMM activity recognition

Individual project. I recorded phone accelerometer and gyroscope data (standing, walking, jumping, still) and built a small Gaussian HMM in NumPy to label the activity over time.

## What’s in here

- `data/raw` – labelled training CSVs  
- `data/test` – held-out recordings for evaluation  
- `notebooks/hmm_activity_recognition.ipynb` – main notebook  
- `src/` – feature code, HMM (Baum-Welch + Viterbi), helpers  
- `report/Formative2_HMM_Report.docx` – write-up for LMS  

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

My Sensor Logger rate was about 100 Hz on an SM-A325N. In code I resample to 50 Hz, then use 1 s windows with 0.5 s hop.

## Methods (short)

- Baum-Welch fits the HMM and stops when the log-likelihood barely changes  
- Viterbi gives the predicted activity path used for the metrics table  

Repo: https://github.com/Emmanuel-NS/hmm-human-activity-recognition
