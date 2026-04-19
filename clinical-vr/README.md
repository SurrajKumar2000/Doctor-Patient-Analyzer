# ClinicalVR — Patient Communication Trainer

## Run (mic works perfectly)

```bash
bash start.sh
```

Then open Chrome: `https://localhost:8443/ClinicalVR.html`

## First time setup
`start.sh` automatically installs mkcert and creates an HTTPS certificate.
Requires: macOS with Homebrew.

## Features
- LLM-generated patient with hidden clinical problem
- Real-time inner monologue shown to doctor
- Trust bar, tension bar, live feedback
- Calgary-Cambridge + TOPSIS debrief scoring
- Voice input (mic) as primary input
- Groq API for patient brain + TTS voice
- Three.js 3D hospital room + patient avatar

## API Key
Get a free Groq key at: https://console.groq.com
Paste it in the config screen before generating a patient.
