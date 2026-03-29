# 🩺 Doctor-Patient Communication Analyzer

> **Multi-modal AI framework for empirical scoring of clinical communication quality**
> Built as a research prototype for PhD-level assessment in Human-Computer Interaction

---

## Overview

This tool analyzes doctor-patient consultation videos using computer vision, speech processing, and large language models to produce an **empirically grounded communication score** — not an AI opinion.

The scoring pipeline is anchored to two peer-reviewed frameworks:

- **Calgary-Cambridge Communication Guide** *(Kurtz & Silverman, 1996)* — defines what to measure and clinical dimension weights
- **TOPSIS Multi-Criteria Decision Analysis** *(Hwang & Yoon, 1981)* — defines how to score empirically using distance from ideal/worst-case vectors

The LLM (Llama 3.3 70B) **explains** the score — it cannot change it.

---

## Architecture

```
YouTube URL
    ↓
yt-dlp          → video download
    ↓
FFmpeg          → audio.mp3 (Groq Whisper)
                → audio.wav (PyAnnote — uncompressed, voice fingerprints intact)
    ↓
┌─────────────────────────────────┬──────────────────────────────────┐
│  VISUAL STREAM                  │  AUDIO STREAM                    │
│  OpenCV + MediaPipe             │  Groq Whisper + PyAnnote 3.1     │
│  • Eye contact estimation       │  • Speech transcription (MP3)    │
│  • Head pose (pitch/yaw)        │  • Speaker diarization (WAV)     │
│  • Facial expression            │  • 3-strategy timestamp merge    │
│  • Nodding detection            │  • 4-signal doctor identification │
│  • Gaze-away timestamps         │  • WPM, fillers, turn latency    │
└─────────────────┬───────────────┴──────────────┬───────────────────┘
                  └──────────────┬────────────────┘
                                 ↓
                  Calgary-Cambridge + TOPSIS scoring
                  (empirical — LLM cannot override)
                                 ↓
                  Groq Llama 3.3 70B
                  (clinical coaching, score explanation)
                                 ↓
                  Streamlit — Quick / Research view
```

---

## Scoring Framework

### Calgary-Cambridge Dimensions

| Dimension | Metric | Weight | Rationale |
|-----------|--------|--------|-----------|
| Rapport Building | Eye contact % | 25% | Primary non-verbal trust signal |
| Gathering Information | Turn balance | 25% | Reflects patient-centred listening |
| Information Giving | Filler word count | 20% | Clarity and confidence of explanation |
| Non-Verbal Communication | Head nodding events | 15% | Active engagement signal |
| Initiating Session | Patient response latency | 15% | Comfort and psychological safety |

### TOPSIS Algorithm

For each dimension, a normalized score is computed relative to an ideal doctor (best case) and an anti-ideal doctor (worst case):

```
normalized  = |value − worst| / |ideal − worst| × 100
d+          = √Σ (weight × (normalized − 100))²   # distance from ideal
d−          = √Σ (weight × normalized)²            # distance from worst
final score = d− / (d+ + d−) × 100
```

The LLM score is **locked** to the TOPSIS output:
```python
feedback["overall_score"] = topsis["topsis_score"]  # immutable
```

---

## Speaker Diarization Pipeline

### Why WAV not MP3 for PyAnnote?
MP3 uses lossy compression that degrades voice frequency data. PyAnnote performs voice fingerprinting on raw acoustic features — MP3 artifacts cause random speaker assignments. WAV preserves these fingerprints.

```
audio.mp3 → Groq Whisper   (transcription — compression acceptable)
audio.wav → PyAnnote 3.1   (diarization  — uncompressed required)
```

### 3-Strategy Timestamp Merge
```
Strategy 1: Maximum overlap  — primary method
Strategy 2: Midpoint lookup  — handles boundary gaps
Strategy 3: Nearest segment  — last resort, never fails
```

### 4-Signal Doctor Identification
```
Signal 1: Medical vocabulary score  (symptom, diagnosis, prescription...)
Signal 2: Question count            (diagnostic questioning pattern)
Signal 3: First speaker             (doctor initiates consultation)
Signal 4: Word count                (doctor typically speaks more)
→ Majority vote across 4 signals
```

---

## Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Video download | yt-dlp | YouTube acquisition |
| Audio extraction | FFmpeg | MP3 + WAV from video |
| Visual analysis | MediaPipe FaceMesh | 468-point facial landmarks |
| Transcription | Groq Whisper Large V3 | Speech-to-text (cloud, fast) |
| Diarization | PyAnnote 3.1 | Speaker separation (local, MPS) |
| Scoring | TOPSIS + Calgary-Cambridge | Empirical multi-criteria scoring |
| Feedback | Groq Llama 3.3 70B | Clinical coaching language |
| Frontend | Streamlit | Quick + Research view modes |

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd doctor-patient-analyzer

pip install -r requirements.txt
```

### 2. API keys (both free)

| Key | Where to get |
|-----|-------------|
| Groq API key | [console.groq.com](https://console.groq.com) |
| HuggingFace token | [huggingface.co](https://huggingface.co) → Settings → Access Tokens |

For HuggingFace, accept model terms at:
- `huggingface.co/pyannote/speaker-diarization-3.1`
- `huggingface.co/pyannote/segmentation-3.0`

### 3. Run

```bash
streamlit run app.py
```

Enter both keys in the sidebar, paste a YouTube URL, click Analyze.

---

## Requirements

```
streamlit
opencv-python
mediapipe==0.10.7
protobuf==3.20.3
numpy
requests
plotly
yt-dlp
torch
torchvision
torchaudio
pyannote.audio>=3.1.0
```

### Mac M1 note
PyAnnote automatically uses **MPS** (Metal Performance Shaders) for Apple Silicon acceleration. No configuration needed — device is auto-detected at runtime.

```python
if torch.backends.mps.is_available():
    pipeline = pipeline.to(torch.device("mps"))
```

### Python version
Requires **Python 3.11**. MediaPipe is not compatible with Python 3.12 due to protobuf API changes.

```bash
conda create -n docpat python=3.11 -y
conda activate docpat
pip install -r requirements.txt
```

---

## View Modes

### 👨‍⚕️ Quick Feedback
- Overall score (0–100) with grade
- Per-dimension score bars
- Radar chart
- Top 3 priority actions
- Strengths

### 🔬 Research Mode
- Full TOPSIS breakdown (d+, d−, weights, normalized values)
- All 7 metric cards
- Calgary-Cambridge radar chart
- Doctor and patient speech statistics
- Response latency timeline
- Visual analysis (expressions, gaze-away, nodding)
- Detailed AI feedback per dimension
- Behavioural insights
- Raw JSON export

---

## Version History

| Version | Key Changes |
|---------|------------|
| v1 | Heuristic diarization (question marks + silence gaps), basic scoring |
| v2 | MediaPipe integration, TOPSIS + Calgary-Cambridge scoring, Groq Llama feedback, Quick/Research modes |
| v3 | PyAnnote 3.1 diarization, WAV audio for voice fingerprinting, 3-strategy merge, 4-signal speaker detection, manual swap button, Mac M1 MPS acceleration |

---

## Known Limitations

- Accuracy depends on audio quality — background noise reduces diarization performance
- Works best with exactly 2 speakers; more speakers may cause label drift
- Eye contact estimation assumes frontal or near-frontal face orientation
- First PyAnnote run downloads ~1GB model to `~/.cache/huggingface`
- MediaPipe requires Python 3.11 — not yet compatible with 3.12

---

## References

- Kurtz, S. & Silverman, J. (1996). *The Calgary-Cambridge Referenced Observation Guides.* Medical Education, 30(2), 83–89.
- Hwang, C.L. & Yoon, K. (1981). *Multiple Attribute Decision Making.* Springer-Verlag.
- Bredin, H. et al. (2021). *End-to-end speaker segmentation for overlap-aware resegmentation.* Interspeech 2021. (PyAnnote)
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* (Whisper)

---

## Author

**Surrajkumar Prabhu Venkatesh**
UX Researcher · MS Computer Science, Cal State Fullerton
