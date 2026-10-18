# Demo Video Recording Guide
# AI_A2_2526883_DEMO.mp4 (max 2 minutes, continuous, unedited)

## Recording Checklist
- [ ] Handwritten card with: Full Name, Registration Number (25/26883), AI-A2-6883
- [ ] Terminal clock visible at start or end
- [ ] Final commit hash shown (`git rev-parse HEAD`)
- [ ] Voice narration identifying yourself

## Script Outline (2 minutes)

### 1. Identity Card (10 sec)
- Show handwritten card to camera
- State: "I am [Your Name], registration 25/26883, verification code AI-A2-6883"
- Show terminal with `git rev-parse HEAD`

### 2. Run Pipeline (30 sec)
```bash
python run_all.py --student-id your_email@ines.ac.rw --seed 6883
```
- Show recreated fingerprint in output
- Show generated artifact list

### 3. Metrics & Plot (30 sec)
```bash
type artifacts\regression_metrics.json
```
- Open one JSON (e.g., regression_metrics.json)
- Open one plot (e.g., artifacts\plots\residuals.png)
- Explain ONE result in your own words (not reading)

### 4. CLI Prediction (20 sec)
```bash
python predict.py --json samples\valid.json
python predict.py --json samples\invalid.json
```
- Show valid output with verification code
- Show invalid output (error)

### 5. FastAPI (20 sec)
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
- Open browser to http://127.0.0.1:8000/docs
- Call /health
- Call /predict with valid input
- Show one validation error (422)

### 6. Tests & Wrap Up (20 sec)
```bash
pytest -q
```
- Show test results passing
- Show declaration, test log files
- Show terminal clock

## Tips
- Practice once before recording
- Keep terminal window large and readable
- Speak clearly and naturally
- Don't read from script - explain in your own words
- MP4 format, max 2 minutes