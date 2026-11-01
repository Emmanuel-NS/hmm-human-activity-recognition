# Finalise now — videos later

Student: **Emmanuel Nsabagasa** (`e.nsabagasa@alustudent.com`)

## Done
- [x] Real Sensor Logger data imported (80 train + 8 test CSVs)
- [x] HMM notebook (Baum–Welch + Viterbi) with outputs
- [x] Word report with figures + metrics (`report/Formative2_HMM_Report.docx`)
- [x] Individual student name filled in report
- [x] Sampling rate documented (SM-A325N ≈ 100 Hz → 50 Hz)

## Do next (before videos)

### 1) Create GitHub repo + push
```powershell
cd C:\Users\PC\Documents\hmm-human-activity-recognition
gh auth login
gh repo create hmm-human-activity-recognition --private --source=. --remote=origin --push
```
Or in the browser: GitHub → New repository → then:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/hmm-human-activity-recognition.git
git push -u origin main
```
Paste the repo URL into the Word report where it says `[PASTE REPO URL AFTER PUSH]`.

### 2) Quick proof-read the docx
Open `report\Formative2_HMM_Report.docx` — check name, use case, figures, metrics table. Leave the four `[PASTE LINK]` video lines empty for now.

## Later (when videos are ready)
1. Upload 4 videos (standing / walking / jumping / still) — face + full body + phone.
2. Paste links into the report.
3. Re-save docx → upload to LMS + confirm GitHub link.

## Optional polish
If you want a stronger standing-vs-walking score before submit, say so and we can tune features/windowing.
