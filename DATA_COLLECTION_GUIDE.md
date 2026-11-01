# Data collection guide (do this before final submission)

The repo currently includes **synthetic** CSVs so the notebook runs. Graders expect **your real Sensor Logger files** and **4 identity videos**.

## Rubric targets

- ≥ **50** well-labelled cleaned files (5–10 s each)
- **4 activities**, each totaling ≥ **1 min 30 s**
- Timestamp column in every CSV
- Sampling rate listed **per group member**
- **4 videos**: full body + face visible + phone + good lighting (one per activity)

## Steps

1. Install Sensor Logger → enable Accelerometer + Gyroscope → note Hz.
2. For each activity, record 12–15 takes of 8–10 s (waist / pocket consistent).
3. Export CSV; rename like `walking_03_fs50.csv`.
4. Put most files in `data/raw/<activity>/`.
5. Move **≥2 files** (ideally ≥1 per activity or two full sessions) to `data/test/<activity>/`.
6. Delete synthetic files once real ones are in place.
7. Re-run the notebook; paste figures + metrics table into `report/REPORT_OUTLINE.md` → export Word docx.
8. Host videos (Drive/YouTube unlisted) and paste links; failing videos = **zero** per brief.
9. Push to GitHub; submit doc/docx on LMS.

## Column tip

If Sensor Logger splits accel/gyro into separate files, merge on timestamp (nearest) or export a combined session. The loader accepts names containing `accelerometer_x`, `gyroscope_x`, `seconds_elapsed`, etc.
