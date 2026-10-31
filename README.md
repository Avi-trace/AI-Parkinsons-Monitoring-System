# NeuroTrack AI — AI Parkinson's Monitoring System

Webcam and drawing-based motor screening: finger tapping, rest/postural tremor and
Archimedean spiral drawing, with longitudinal session tracking.

> **Research prototype — not a medical device.** Scores are uncalibrated screening
> indices, not a diagnosis, and have not been validated against clinician
> MDS-UPDRS ratings. Do not use them for clinical decisions.

## Architecture

```
frontend/frontend/        React + Vite
  src/tracking/           shared webcam + MediaPipe Tasks HandLandmarker (GPU), dynamic fps
  src/components/         Finger tap, Tremor, Spiral, Dashboard
backend/                  FastAPI
  neurotrack/core/        signal processing (uniform resampling, Welch PSD, PCA) + quality gates
  neurotrack/analysis/    finger_tap.py, tremor.py, spiral.py
  neurotrack/db/          SQLite repository (same interface can back Postgres/Supabase later)
  neurotrack/api/         request schemas + routes
tests/                    synthetic-signal regression suite + API tests
research/desktop/         standalone OpenCV prototypes (track.py, valley.py) — not used by the app
```

Hand tracking runs **in the browser on the GPU** (WebGL delegate, automatic CPU
fallback). Video never leaves the machine; only landmark-derived numbers are sent
to the backend.

## Run it locally (Windows, NVIDIA laptop GPU)

Requirements: Python 3.11+, Node 20+, Chrome or Edge.

**1. Make the browser use the RTX GPU, not the integrated Intel GPU.**
Windows Settings → System → Display → Graphics → add Chrome (or Edge) → Options →
*High performance* (NVIDIA). Restart the browser. Check `chrome://gpu`: WebGL should
be "Hardware accelerated" and the renderer should mention NVIDIA.

**2. Backend**

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn neurotrack.main:app --reload --port 5000
```

API docs: http://localhost:5000/docs

**3. Frontend** (second terminal)

```powershell
cd frontend\frontend
npm install        # also copies the MediaPipe wasm and downloads the hand model into public/
npm run dev
```

Open http://localhost:5173. If the browser picks the laptop's built-in camera instead
of your external webcam, click the camera icon in the address bar and choose it there.

While a test is open the camera badge shows the measured **fps** and whether the
tracker is on the **GPU**. The tracker asks the webcam for 60 fps and steps capture
resolution down or up (1280×720 → 480×360) to keep inference within the frame budget.
Resolution is frozen while a test records.

**4. Tests**

```powershell
pytest            # from the repo root, with the backend venv active
```

## What the backend measures

| Test | Key outputs | Rejected (HTTP 422) when |
|---|---|---|
| Finger tap (`POST /api/finger-tap`) | tap rate, amplitude (÷ hand size), first-vs-last-third decrement, opening/closing speed, rhythm CV, hesitations | < 20 fps, hand lost > 0.5 s, < 80 % coverage, no taps |
| Tremor (`POST /api/tremor-test`, `mode: rest \| postural`) | dominant frequency, amplitude in approx. cm, MDS-UPDRS-style amplitude grade, PD-band ratio, onset latency, asymmetry | < 15 fps or poor tracking for every hand |
| Spiral (`POST /api/spiral-test`) | deviation from the person's own fitted spiral, loop-spacing regularity, micrographia index, radial tremor frequency/amplitude | < 50 points, < 2 s, < 1.5 turns |

A recording that is too poor to analyse is **never** reported as "Normal". Every
result includes a `quality` block (measured fps, coverage, gaps, warnings) and an
`algorithm_version`.

Sessions (`POST /api/session`) link server-stored results by `result_id` and record
medication state (on/off/unknown) and minutes since the last dose. The session's
**motor screening index** is the mean over tests of the worst result per test.
`composite`, `risk_level` and `*_score` are kept as deprecated aliases for the current
dashboard.

## Configuration

Settings come from environment variables; see `backend/.env.example`. In production,
set `NEUROTRACK_ENV=production` and `NEUROTRACK_API_KEYS` (clients send `X-API-Key`);
the server refuses to start without keys and disables `/docs`. Serve the frontend over
HTTPS: browsers only allow camera access on HTTPS or localhost.

## Data and privacy

- Patient IDs only; no names or contact details are stored. Local data lives in
  `backend/data/` (git-ignored). Never commit it.
- Large signal arrays are returned to the browser for charts but not persisted.
- Before any real-patient use you still need informed consent, login with per-patient
  access control, encryption at rest, audit logging and data deletion (India's DPDP
  Act 2023; HIPAA/GDPR where applicable).

## Known limitations / next steps

- **Clinical validation:** thresholds are placeholders (`backend/neurotrack/core/constants.py`).
  Calibrate against clinician-rated recordings, then report sensitivity/specificity and
  test-retest reliability.
- **Frontend rework (planned):** guided assessment flow, rest-tremor mode and left/right
  hand selection in the UI, camera picker, medication-state entry, accessibility,
  plain-language patient view vs clinician view.
- **Database:** SQLite for now; move to Postgres/Supabase by implementing
  `neurotrack/db/repository.py`'s interface.
- **Higher-rate tremor sensing:** the planned ESP32 + MPU6050 wrist band (100+ Hz) removes
  the webcam frame-rate ceiling on tremor frequency.
