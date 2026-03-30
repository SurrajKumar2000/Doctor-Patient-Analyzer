# 🩺 Doctor-Patient Communication Analyzer

> Multi-modal AI framework for empirical scoring of clinical communication quality.
> Built as a PhD-level HCI research prototype — Saint Louis University, Prof Min Choi.

---

## What It Does

Analyzes a YouTube doctor-patient consultation video and produces an **empirically grounded communication score** — not an AI opinion. The LLM explains the score. It cannot change it.

```
YouTube URL → video → visual stream + audio stream → TOPSIS score → LLM coaching
```

---

## Scoring Philosophy

Two peer-reviewed frameworks anchor every score:

| Framework | Role |
|-----------|------|
| **Calgary-Cambridge** (Kurtz & Silverman, 1996) | Defines *what* to measure and clinical weights |
| **TOPSIS** (Hwang & Yoon, 1981) | Defines *how* to score using distance from ideal/worst vectors |

```
normalized  = |value − worst| / |ideal − worst| × 100
d+          = √Σ (weight × (normalized − 100))²   ← distance from ideal doctor
d−          = √Σ (weight × normalized)²            ← distance from worst doctor
final score = d− / (d+ + d−) × 100
```

The score is computed before the LLM is called. `feedback["overall_score"]` is overwritten to `topsis["topsis_score"]` after every LLM response — immutably.

---

## Version History

### v1 — Foundation
**Stack:** Groq Whisper + heuristic diarization + basic scoring

- Groq Whisper Large V3 for speech-to-text via API
- Heuristic diarization: speaker switches guessed from question marks and silence gaps
- ~60% speaker accuracy
- Simple metric display

**Limitation:** Diarization was unreliable. "?" detection misses context entirely.

---

### v2 — Visual Layer + Scoring Framework
**Added:** MediaPipe FaceMesh + TOPSIS + Calgary-Cambridge + dual view modes

- OpenCV + MediaPipe 468-point facial landmark analysis
- Eye contact estimation, head pose (pitch/yaw), nodding detection, gaze-away timestamps
- TOPSIS multi-criteria scoring anchored to Calgary-Cambridge dimensions
- Groq Llama 3.3 70B for clinical coaching feedback
- **Quick Feedback** mode: score + top 3 actions
- **Research Mode**: full TOPSIS breakdown, radar chart, per-dimension feedback

**Limitation:** Still using heuristic diarization. MP3 audio sent to both Whisper and diarizer.

---

### v3 — Accurate Diarization
**Added:** PyAnnote 3.1 + WAV audio + 3-strategy merge + 4-signal speaker detection

**The core fix:**
```
Before:  video → audio.mp3 → Whisper ✅ and PyAnnote ❌
After:   video → audio.mp3 → Whisper ✅  (transcription — compression fine)
               → audio.wav → PyAnnote ✅  (diarization — uncompressed PCM required)
```

MP3 lossy compression degrades the voice frequency data PyAnnote uses for speaker fingerprinting. WAV preserves raw acoustic fingerprints → ~94% diarization accuracy vs ~60%.

**3-Strategy Timestamp Merge** (Whisper timestamps ↔ PyAnnote timestamps):
1. Maximum overlap — primary method
2. Midpoint lookup — handles boundary gaps
3. Nearest neighbour — last resort, never fails

**4-Signal Doctor Identification:**
1. Medical vocabulary score
2. Question count
3. First speaker (doctor initiates)
4. Word count (doctor speaks more)

→ Majority vote across 4 signals. Manual swap button for when auto-detection is wrong.
Mac M1 MPS acceleration — PyAnnote auto-detects Apple Silicon.

---

### v4 — Brief Coverage + Score Consistency Fix
**Added:** 5 new metrics + session state fix + transcript removed

#### Bug Fix: Score Mismatch Between Modes
**Root cause:** Switching Quick ↔ Research mode triggers a Streamlit script rerun. The `analyze_btn` resets to `False` → analysis block skips → different variables in scope → different renders.

**Fix:** All results stored in `st.session_state["analysis_results"]` after analysis. Both modes read from the same cached dict. `OVERALL_SCORE` is set once from session state and displayed in a **shared banner before the mode split** — one piece of HTML, physically impossible to differ.

```python
OVERALL_SCORE = topsis["topsis_score"]   # set once from session_state

# Shared banner — rendered BEFORE is_quick branch
# Both modes see the same number
st.markdown(f'... {OVERALL_SCORE}/100 ...')

if is_quick:
    ...  # no score card here
else:
    ...  # no score card here
```

#### 5 New Metrics (covering brief gaps)

| Metric | Brief Task | What It Measures |
|--------|-----------|-----------------|
| **Empathy Score** | Task 4: tone alignment | Groq Llama rates doctor warmth 0–100 |
| **Patient Engagement** | Task 4: elaboration rate | Words/turn, voluntary questions, elaboration ratio |
| **Hesitation Windows** | Task 2: response latency as confusion signal | Patient pauses ≥ 2s flagged with severity + context |
| **Brow Furrow Index** | Task 1: brow furrow as confusion signal | % frames with furrowed brows from MediaPipe |
| **Session Arc** | Task 5: improvement across turns | First half vs second half patient speech ratio |

**Transcript removed** — diarization errors made it unreliable and distracting from scores.

---

## Architecture

> 📄 **[View Full Architecture Diagram → architecture_diagram.pdf](./architecture_diagram.pdf)**

```
YouTube URL
    ↓
yt-dlp → video download
    ↓
FFmpeg ──────────────────────────────────────────────┐
  audio.mp3 → Groq Whisper (transcription)           │
  audio.wav → PyAnnote 3.1 (diarization, MPS)        │  video → OpenCV → frames[]
    ↓                   ↓                             ↓
 text + timestamps   speaker labels            MediaPipe FaceMesh
         └──────────────┘                      (eye contact, pose,
     3-strategy timestamp merge                 expression, nodding)
     4-signal doctor identification                    │
              ↓                                        │
       speaker_stats{}  ←──────────────────────────────┘
              ↓                        ↑
       Calgary-Cambridge        visual_stats{}
       (5 weighted dims)
              ↓
       TOPSIS Algorithm  →  score = d− / (d+ + d−) × 100
              ↓
       OVERALL_SCORE [LOCKED — LLM cannot change]
              ↓
       Groq Llama 3.3 70B → coaching + empathy score
              ↓
       Streamlit — Quick / Research view modes
       (session_state cache — score identical in both)
```

*Fig. 1 — System Architecture: Calgary-Cambridge Grounded TOPSIS Multimodal Clinical Communication Scoring Pipeline. See [`architecture_diagram.pdf`](./architecture_diagram.pdf) for the full annotated diagram.*

---

## Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Video download | yt-dlp | YouTube acquisition |
| Audio extraction | FFmpeg | MP3 + WAV from video |
| Visual analysis | MediaPipe FaceMesh | 468-point facial landmarks |
| Transcription | Groq Whisper Large V3 | Speech-to-text (cloud) |
| Diarization | PyAnnote 3.1 | Speaker separation (local, MPS) |
| Scoring | TOPSIS + Calgary-Cambridge | Empirical multi-criteria scoring |
| Feedback + Empathy | Groq Llama 3.3 70B | Clinical coaching language |
| Frontend | Streamlit | Quick + Research view modes |

---

## Setup

### Requirements

```
Python == 3.11        (not 3.12 — MediaPipe incompatible)
protobuf == 3.20.3    (not 4.x — breaks MediaPipe MessageFactory)
```

### Install

```bash
conda create -n docpat python=3.11 -y
conda activate docpat

pip install protobuf==3.20.3        # must install FIRST
pip install mediapipe==0.10.7
pip install opencv-python streamlit numpy requests plotly yt-dlp
pip install torch torchvision torchaudio
pip install pyannote.audio
```

### Verify before running

```bash
python -c "
import sys, google.protobuf, mediapipe, torch
print('Python:   ', sys.version[:6])                   # must be 3.11.x
print('Protobuf: ', google.protobuf.__version__)        # must be 3.20.3
print('MediaPipe:', mediapipe.__version__)              # 0.10.7
print('MPS:      ', torch.backends.mps.is_available())  # True on M1

fm = mediapipe.solutions.face_mesh.FaceMesh()
fm.close()
print('FaceMesh: OK ✅')
"
```

### Run

```bash
conda activate docpat
streamlit run app_v4.py
```

---

## API Keys

Both are free.

| Key | Source | Used For |
|-----|--------|---------|
| Groq API `gsk_...` | [console.groq.com](https://console.groq.com) | Whisper transcription + Llama feedback |
| HuggingFace `hf_...` | [huggingface.co](https://huggingface.co) → Settings → Tokens | PyAnnote diarization (local) |

Accept model terms at both:
- `huggingface.co/pyannote/speaker-diarization-3.1`
- `huggingface.co/pyannote/segmentation-3.0`

First PyAnnote run downloads ~1GB to `~/.cache/huggingface`.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `'MessageFactory' object has no attribute 'GetPrototype'` | protobuf 4.x/5.x | `pip install protobuf==3.20.3 --force-reinstall` |
| `ValidatedGraphConfig Initialization failed` | Python 3.12 | `conda activate docpat` (Python 3.11) |
| PyAnnote gives wrong speaker labels | Auto-detection failed | Click **🔄 Swap Doctor ↔ Patient** in sidebar |
| Quick / Research modes show different scores | Missing session state | Use v4 — score cached once, shared across modes |

---

## Calgary-Cambridge Dimensions

| Dimension | Metric | Weight |
|-----------|--------|--------|
| Rapport Building | Eye contact % | 25% |
| Gathering Information | Turn balance | 25% |
| Information Giving | Filler word count | 20% |
| Non-Verbal Communication | Nodding events | 15% |
| Initiating Session | Patient response latency | 15% |

---

## View Modes

**👨‍⚕️ Quick Feedback**
Overall score · Empathy score · Per-dimension bars · Radar chart · Top 3 actions · Strengths

**🔬 Research Mode**
TOPSIS d+/d− breakdown · Empathy assessment · Patient engagement panel · Hesitation windows · Brow furrow index · Session arc · Visual analysis · Per-dimension AI feedback · Raw JSON export

---

## References

- Kurtz, S. & Silverman, J. (1996). The Calgary-Cambridge Referenced Observation Guides. *Medical Education*, 30(2), 83–89.
- Hwang, C.L. & Yoon, K. (1981). *Multiple Attribute Decision Making*. Springer-Verlag.
- Bredin, H. et al. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. *Interspeech 2021*.
- Radford, A. et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *(Whisper)*

---

*v4 · Groq Whisper (MP3) + PyAnnote (WAV · Mac M1 MPS) + Groq Llama 3.3 70B · TOPSIS + Calgary-Cambridge*
