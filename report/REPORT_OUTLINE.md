# Modeling Human Activity States Using Hidden Markov Models

**Course / Formative 2**  
**Group members:** [Name 1], [Name 2], …  
**GitHub repository:** [paste URL]  
**Date:** [fill]

---

## 1. Background and Motivation (≈½ page)

Human activity recognition (HAR) from wearable and smartphone inertial sensors supports applications such as fall detection, fitness tracking, and context-aware smart homes. True activity labels are **hidden**; only noisy accelerometer and gyroscope streams are observed. Hidden Markov Models (HMMs) are a natural fit because they model (i) temporal persistence of an activity via transition probabilities and (ii) noisy sensor patterns via emission distributions.

**Our group’s unique use case:**  
*[Example — rewrite in your own words]*  
We target **short-burst indoor activity logging for a student wellness / sedentary-break reminder**: detecting whether a user is still/standing at a desk versus walking or jumping during a movement break, using only the phone already in their pocket or hand. Reliable transitions between still/standing and locomotion are needed so reminders are not triggered during walking, and so “active break” compliance can be verified without cameras.

---

## 2. Data Collection and Pre-processing

### 2.1 Collection protocol

| Activity | Duration per take | Notes |
|----------|-------------------|-------|
| Standing | 5–10 s | Phone steady at waist |
| Walking | 5–10 s | Consistent pace |
| Jumping | 5–10 s | Continuous jumps |
| Still | 5–10 s | Phone flat on surface |

- App used: Sensor Logger / [other]  
- Sensors: accelerometer (x,y,z), gyroscope (x,y,z), timestamp  
- Target: ≈50 labelled files; ≥1 min 30 s total per activity  

### 2.2 Member sampling rates (required)

| Member | Phone | App setting / measured rate |
|--------|-------|-----------------------------|
| [Name] | [model] | [e.g. 50 Hz] |
| [Name] | [model] | [e.g. 100 Hz] |

**Harmonization:** all recordings are linearly resampled to a common **50 Hz** grid before windowing so features are comparable across phones.

### 2.3 Windowing logic

At 50 Hz we use **1.0 s windows (50 samples)** with **0.5 s hop**.  
FFT frequency resolution ≈ 1 Hz, which is enough to separate still/standing (near 0 Hz) from walking cadence (~1.5–2 Hz) and jumping (~2–3 Hz), while remaining short enough to track activity changes.

### 2.4 Video evidence (mandatory — zero if missing)

Insert / link **four** videos (face visible, full body, phone in hand, good lighting):

1. Standing: https://youtube.com/shorts/fWiTltw83o8  
2. Walking: https://youtube.com/shorts/m0d-yOL5tSE  
3. Jumping: https://youtube.com/shorts/JhfENFsJmo8  
4. Still: https://youtube.com/shorts/YM_wLnQTzg0  

### 2.5 Sample visualizations

*[Paste the 2×2 accelerometer-magnitude plots from the notebook]*

**Figure 1.** Example raw |a| traces after resampling for each activity.

---

## 3. HMM Setup and Implementation

### 3.1 Model components

| Element | Definition in this project |
|---------|----------------------------|
| Hidden states Z | standing, walking, jumping, still |
| Observations X | 10-D feature vector per window (time + frequency) |
| A | 4×4 transition matrix |
| B | Diagonal Gaussian emission N(x \| μ_i, σ²_i) |
| π | Initial state distribution |

### 3.2 Features (>2 time-domain, ≥1 FFT)

Time: mean/var/std/RMS of accel magnitude, SMA, corr(x,y), gyro mean/std.  
Frequency: dominant frequency and spectral energy from FFT of accel magnitude.  
**Normalization:** Z-score fit on training only — equalizes units before Gaussian emissions.

### 3.3 Algorithms (brief, tied to metrics)

- **Baum–Welch** maximizes sequence log-likelihood with stopping criterion `|Δll| < 1e-4`, learning A/B/π that support generalization.  
- **Viterbi** produces a discrete MAP state path; those hard labels are used to compute **sensitivity, specificity, accuracy**, and the **confusion matrix**.  
- Soft Forward–Backward posteriors are used only inside EM, not for the rubric table.

Implementation: NumPy Gaussian HMM in `src/hmm.py`, driven by `notebooks/hmm_activity_recognition.ipynb`.

---

## 4. Results and Interpretation

### 4.1 Transition matrix

*[Paste heatmap of A]*  
**Figure 2.** Learned transition probabilities. High diagonal mass indicates realistic “stay in the same activity” behaviour for 1 s windows.

### 4.2 Emission parameters

*[Paste emission mean / variance heatmaps]*  
**Figure 3.** State-conditional feature means/variances in Z-scored space.

### 4.3 Decoded sequences

*[Paste 1–2 Viterbi plots]*  
**Figure 4.** True vs Viterbi-predicted activity over windows.

### 4.4 Unseen-data evaluation

**How unseen data were obtained:** *[e.g. new recording session on a different day / different room / different group member]*  

**Generalization:** *[Does accuracy hold? Which states confuse?]*  

| State (Activity) | Number of Samples | Sensitivity | Specificity | Overall Accuracy |
|------------------|-------------------|-------------|-------------|------------------|
| standing | | | | |
| walking | | | | |
| jumping | | | | |
| still | | | | |

*[Paste confusion matrix figure]*  
**Figure 5.** Confusion matrix on held-out test files.

---

## 5. Discussion and Conclusion

- **Easiest / hardest:** *[e.g. jumping vs still easy; standing vs still harder due to similar low motion]*  
- **Transitions:** *[diagonal-heavy A matches short continuous takes]*  
- **Noise / sampling:** *[resampling 100→50 Hz; sensor bias; phone orientation]*  
- **Improvements:** more participants, magnetometer, full-covariance Gaussians, longer continuous multi-activity sessions for richer A.  

**Conclusion:** A Gaussian HMM with Baum–Welch training and Viterbi decoding can recover activity states from phone IMU features; performance on unseen takes *[summarize]*. This supports our *[use case]* by distinguishing sedentary vs active phone contexts without cameras.

---

## Appendix (optional, keep report body 4–5 pages)

- Link to GitHub  
- File naming convention: `{activity}_{id}_fs{Hz}.csv`  
- Exact package versions from `requirements.txt`
