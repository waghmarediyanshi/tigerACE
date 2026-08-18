# TigerID AI — Innovative Hackathon Prototype

## Innovations
1. Explainable AI: returns stripe landmarks, intersections, distinctive gaps and human-readable match reasons.
2. Active Learning: forest-officer verification is stored as labelled feedback and increases verified training examples.

## Run
Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Then serve the frontend:
```bash
python -m http.server 5500 --directory frontend
```
Open http://127.0.0.1:5500

## Demo
Upload an image -> Identify Tiger -> show confidence and stripe evidence -> click "Correct — Learn" -> show verified feedback count increasing.

## Important
The current matcher is a lightweight deterministic visual-feature engine so the prototype runs without a GPU. Replace the matching section with a trained tiger re-identification model for production.
Recommended production pipeline: tiger detector -> flank detector -> re-ID embedding -> vector similarity search -> human-reviewed active-learning queue.


## New visual innovation upgrade

The identification result now shows two evidence panels:
- Uploaded camera-trap image with numbered orange stripe landmarks and connecting lines.
- Stylized catalogue-reference flank with the corresponding numbered landmarks.

This makes the Explainable AI feature visible during the pitch. The catalogue panel is intentionally marked as a prototype visualization; production should use the actual stored reference flank image for the matched tiger.
