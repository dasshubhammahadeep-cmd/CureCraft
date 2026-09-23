# CureCraft AI Enhanced — VS Code Run Guide

## Included
- Existing Patient, Doctor, Admin portals and central case flow preserved.
- One unified symptom-analysis service replaces the duplicate hard-coded simulator logic.
- Optional live OpenAI Responses API + web search integration.
- Deterministic emergency safety screen remains active even when the AI service is unavailable.
- AI explicitly avoids individualized medication prescribing; the doctor remains the prescription authority through the existing medicine database.
- Persistent doctor reviews/ratings stored in `curecraft.sqlite3` (SQLite, created automatically).
- Patient sees each doctor's aggregate star rating before appointment selection.
- Admin sees aggregate ratings and recent reviews.
- Multiple attachments in any file format with per-file remove and Remove all controls.
- 25 MB per-file and 60 MB per-submission safeguards.
- Modern animated UI: scroll progress, cursor glow, ambient particles, reveal motion, interactive stars, live AI state.
- Existing FHIR-style case observation is retained in the AI report and patient case report.

## Install
```powershell
python -m pip install -r requirements.txt
```

## Enable live web-grounded AI (recommended for the hackathon demo)
PowerShell:
```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:CURECRAFT_AI_MODEL="gpt-5.6"
python CureCraft_AI_Enhanced.py
```

Command Prompt:
```bat
set OPENAI_API_KEY=YOUR_API_KEY
set CURECRAFT_AI_MODEL=gpt-5.6
python CureCraft_AI_Enhanced.py
```

The app is served at `http://127.0.0.1:8000`.

## Demo accounts from the supplied codebase
- Admin: `admin_chief` / `admin`
- Doctor: `DOC-8831` / `doc`
- Doctor: `DOC-9924` / `doc`
- Patient: `ABHA-9844` / `pass`

## Important prototype note
This remains a hackathon prototype, not a production clinical system. Keep human clinical review in the loop and do not use the AI result as a diagnosis or as an automatic prescription. The live web-grounding uses OpenAI's Responses API `web_search` tool as documented in the current API documentation.
