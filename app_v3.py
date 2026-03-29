import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
import time
import subprocess
import requests
import torch
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Doctor-Patient Communication Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 50%, #2c7a7b 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
    color: white; box-shadow: 0 8px 32px rgba(26,54,93,0.3);
}
.main-header h1 { margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.5px; }
.main-header p  { margin:0.5rem 0 0; opacity:0.85; font-size:1rem; }

.metric-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center; height: 100%;
}
.metric-card .label { font-size:.8rem; color:#718096; font-weight:500;
                       text-transform:uppercase; letter-spacing:.5px; }
.metric-card .value { font-size:2rem; font-weight:700; margin:.4rem 0 .2rem; }
.metric-card .sub   { font-size:.8rem; color:#a0aec0; }

.score-excellent { color:#38a169; }
.score-good      { color:#3182ce; }
.score-fair      { color:#d69e2e; }
.score-poor      { color:#e53e3e; }

.quick-score-card {
    background: white; border-radius: 16px; padding: 2rem;
    border: 1px solid #e2e8f0; box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    text-align: center; margin-bottom: 1.5rem;
}
.quick-score-card .big-score { font-size: 4rem; font-weight: 800; line-height: 1; }
.quick-score-card .grade     { font-size: 1.2rem; font-weight: 600; margin-top: .5rem; }

.feedback-section {
    background: white; border-radius: 12px; padding: 1.5rem;
    border: 1px solid #e2e8f0; margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.topsis-box {
    background: linear-gradient(135deg, #f0fff4, #ebf8ff);
    border: 1px solid #9ae6b4; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    font-size: .88rem; color: #2d3748;
}
.tag { display:inline-block; padding:.25rem .7rem; border-radius:999px;
       font-size:.75rem; font-weight:600; margin:.2rem; }
.tag-green  { background:#c6f6d5; color:#276749; }
.tag-yellow { background:#fefcbf; color:#975a16; }
.tag-red    { background:#fed7d7; color:#9b2c2c; }

.progress-bar-wrap { background:#edf2f7; border-radius:999px; height:10px;
                     margin:.4rem 0 1rem; overflow:hidden; }
.progress-bar-fill { height:100%; border-radius:999px; transition:width .6s ease; }

.timeline-item { border-left:3px solid #4299e1; padding-left:1rem; margin-bottom:1rem; }
.timeline-time { font-size:.75rem; color:#4299e1; font-weight:600; }
.timeline-text { font-size:.85rem; color:#4a5568; margin-top:.2rem; }

.stButton>button {
    background: linear-gradient(135deg,#2b6cb0,#2c7a7b); color: white;
    border: none; padding: .6rem 2rem; border-radius: 8px;
    font-weight: 600; font-size: .95rem; transition: all .2s; width: 100%;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 4px 12px rgba(43,108,176,.4); }

.info-box { background:#ebf8ff; border:1px solid #90cdf4; border-radius:8px;
            padding:1rem 1.2rem; color:#2c5282; font-size:.9rem; }
.warn-box { background:#fffaf0; border:1px solid #f6ad55; border-radius:8px;
            padding:1rem 1.2rem; color:#7b341e; font-size:.9rem; }
.mode-badge {
    display:inline-block; padding:.3rem .9rem; border-radius:999px;
    font-size:.78rem; font-weight:700; letter-spacing:.5px;
    text-transform:uppercase; margin-bottom:1rem;
}
.mode-quick    { background:#c6f6d5; color:#276749; }
.mode-research { background:#e9d8fd; color:#553c9a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
from mediapipe.python.solutions import face_mesh as _mp_face_mesh_mod

class _FaceMeshNS:
    FaceMesh = _mp_face_mesh_mod.FaceMesh

mp_face_mesh = _FaceMeshNS()

LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
NOSE_TIP   = 1;  CHIN = 152
LEFT_EAR   = 234; RIGHT_EAR = 454
LEFT_BROW  = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
UPPER_LIP  = 13;  LOWER_LIP = 14

FILLER_WORDS = [
    "um","uh","umm","uhh","er","err","ah",
    "like","you know","i mean","basically",
    "i don't know","kind of","sort of"
]

# ─────────────────────────────────────────────
# DEVICE DETECTION — Mac M1 uses MPS
# ─────────────────────────────────────────────
def get_device():
    """
    Mac M1/M2/M3 → MPS (Metal Performance Shaders)
    NVIDIA GPU    → CUDA
    Fallback      → CPU
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ─────────────────────────────────────────────
# CALGARY-CAMBRIDGE + TOPSIS SCORING
# ─────────────────────────────────────────────
# Calgary-Cambridge Communication Guide (Kurtz & Silverman, 1996)
# Defines WHAT to measure + clinical weights
# TOPSIS Algorithm (Hwang & Yoon, 1981)
# Defines HOW to score empirically
# ─────────────────────────────────────────────
CRITERIA = {
    "eye_contact"      : {"weight":0.25,"ideal":100.0,"worst":0.0,  "label":"Eye Contact",   "dimension":"Rapport Building",         "icon":"👁️"},
    "speech_clarity"   : {"weight":0.20,"ideal":0.0,  "worst":20.0, "label":"Speech Clarity","dimension":"Information Giving",        "icon":"🎙️"},
    "turn_balance"     : {"weight":0.25,"ideal":50.0, "worst":90.0, "label":"Turn Balance",  "dimension":"Gathering Information",     "icon":"👂"},
    "response_latency" : {"weight":0.15,"ideal":0.5,  "worst":10.0, "label":"Patient Comfort","dimension":"Initiating Session",       "icon":"⏱️"},
    "body_language"    : {"weight":0.15,"ideal":10.0, "worst":0.0,  "label":"Body Language", "dimension":"Non-Verbal Communication", "icon":"🙆"},
}

def calculate_topsis_score(visual_stats, speaker_stats):
    raw = {
        "eye_contact"      : visual_stats.get("eye_contact_pct", 0),
        "speech_clarity"   : speaker_stats.get("doctor", {}).get("filler_count", None),
        "turn_balance"     : speaker_stats.get("doctor_turn_pct", None),
        "response_latency" : speaker_stats.get("avg_response_latency", None),
        "body_language"    : len(visual_stats.get("nod_timestamps", [])),
    }
    neutral = {
        "speech_clarity"  : 10,
        "turn_balance"    : 65,
        "response_latency": 2.0,
    }
    for key in raw:
        if raw[key] is None:
            raw[key] = neutral.get(key, 0)

    normalized = {}
    for key, val in raw.items():
        ideal = CRITERIA[key]["ideal"]; worst = CRITERIA[key]["worst"]
        rang  = abs(ideal - worst)
        normalized[key] = 100.0 if rang == 0 else min(round(abs(val - worst) / rang * 100, 1), 100.0)

    weighted = {k: normalized[k] * CRITERIA[k]["weight"] for k in normalized}
    d_plus   = np.sqrt(sum((CRITERIA[k]["weight"] * (normalized[k] - 100)) ** 2 for k in normalized))
    d_minus  = np.sqrt(sum((CRITERIA[k]["weight"] * normalized[k]) ** 2 for k in normalized))
    topsis_raw  = d_minus / (d_plus + d_minus) if (d_plus + d_minus) > 0 else 0
    final_score = round(topsis_raw * 100, 1)

    grade = ("Excellent" if final_score >= 80 else "Good" if final_score >= 65
             else "Fair" if final_score >= 45 else "Needs Improvement")

    breakdown = {}
    for key in normalized:
        val = normalized[key]
        status = ("✅","score-excellent") if val>=80 else ("✅","score-good") if val>=60 else ("⚠️","score-fair") if val>=40 else ("❌","score-poor")
        breakdown[key] = {
            "raw_value"  : raw[key],      "normalized": normalized[key],
            "weighted"   : round(weighted[key], 2), "weight_pct": f"{int(CRITERIA[key]['weight']*100)}%",
            "dimension"  : CRITERIA[key]["dimension"], "label": CRITERIA[key]["label"],
            "icon"       : CRITERIA[key]["icon"],      "status_icon": status[0], "css_class": status[1],
        }
    return {"topsis_score":final_score,"grade":grade,"d_plus":round(d_plus,4),
            "d_minus":round(d_minus,4),"breakdown":breakdown,"normalized":normalized,"raw":raw}

def radar_chart(topsis_result):
    labels = [CRITERIA[k]["label"] for k in CRITERIA]
    values = [topsis_result["normalized"][k] for k in CRITERIA]
    vc = values + [values[0]]; lc = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vc, theta=lc, fill="toself",
        fillcolor="rgba(43,108,176,0.15)", line=dict(color="#2b6cb0",width=2), name="Doctor Score"))
    ic = [100]*len(labels); icc = ic+[ic[0]]
    fig.add_trace(go.Scatterpolar(r=icc, theta=lc, fill="toself",
        fillcolor="rgba(56,161,105,0.05)", line=dict(color="#38a169",width=1.5,dash="dot"), name="Ideal Doctor"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,100],tickfont=dict(size=10),gridcolor="#e2e8f0"),
                   angularaxis=dict(tickfont=dict(size=11,color="#2d3748")), bgcolor="white"),
        showlegend=True, legend=dict(orientation="h",y=-0.15),
        margin=dict(t=20,b=40,l=40,r=40), paper_bgcolor="white", height=380)
    return fig

# ─────────────────────────────────────────────
# MEDIAPIPE HELPERS
# ─────────────────────────────────────────────
def estimate_eye_contact(lm, w, h):
    nose=lm.landmark[NOSE_TIP]; left=lm.landmark[LEFT_EAR]; right=lm.landmark[RIGHT_EAR]
    top=lm.landmark[10]; chin=lm.landmark[CHIN]
    fw=abs(left.x-right.x); cx=(left.x+right.x)/2; cy=(top.y+chin.y)/2
    ho=abs(cx-0.5)/max(fw,0.01); vo=abs(cy-0.5); yp=abs(nose.x-cx)/max(fw,0.01)
    return float(np.clip(max(0,1-ho*2-vo*1.5-yp*3),0,1))

def estimate_head_pose(lm):
    nose=lm.landmark[NOSE_TIP]; chin=lm.landmark[CHIN]; top=lm.landmark[10]
    left=lm.landmark[LEFT_EAR]; right=lm.landmark[RIGHT_EAR]
    fv=chin.y-top.y; np_=(nose.y-top.y)/max(fv,0.01)
    pitch=("Head tilted up" if np_<0.40 else "Head tilted down" if np_>0.60 else "Head level")
    cx=(left.x+right.x)/2; yo=nose.x-cx
    yaw=("Turned left" if yo<-0.05 else "Turned right" if yo>0.05 else "Facing forward")
    return pitch, yaw

def estimate_engagement(lm):
    lb=np.mean([lm.landmark[i].y for i in LEFT_BROW]); rb=np.mean([lm.landmark[i].y for i in RIGHT_BROW])
    le=np.mean([lm.landmark[i].y for i in LEFT_EYE_INDICES]); re=np.mean([lm.landmark[i].y for i in RIGHT_EYE_INDICES])
    return {"brow_raise":float(((le-lb)+(re-rb))/2),
            "mouth_open":float(lm.landmark[LOWER_LIP].y-lm.landmark[UPPER_LIP].y),
            "brow_spread":float(lm.landmark[336].x-lm.landmark[107].x)}

def classify_expression(eng):
    if eng["mouth_open"]>0.04: return "Speaking / Reacting"
    if eng["brow_raise"]>0.06: return "Attentive / Raised brow"
    if eng["brow_spread"]<0.07: return "Concerned / Furrowed"
    return "Neutral"

def detect_nod(prev_y, curr_y, threshold=0.015):
    return abs(curr_y-prev_y)>threshold

# ─────────────────────────────────────────────
# VIDEO ANALYSIS
# ─────────────────────────────────────────────
def analyze_video(video_path, sample_rate=4, max_frames=500, progress_cb=None):
    cap=cv2.VideoCapture(video_path); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration=total/fps; frames_data=[]; frame_idx=0; sampled=0; prev_nose_y=None; nod_events=[]
    with mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=4,refine_landmarks=False,
                                min_detection_confidence=0.3,min_tracking_confidence=0.3) as face_mesh:
        while cap.isOpened() and sampled<max_frames:
            ret,frame=cap.read()
            if not ret: break
            if frame_idx%sample_rate==0:
                ts=frame_idx/fps; rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=face_mesh.process(rgb)
                entry={"timestamp":round(ts,2),"face_found":False,"num_faces":0}
                if res.multi_face_landmarks:
                    entry["face_found"]=True; entry["num_faces"]=len(res.multi_face_landmarks)
                    lm=res.multi_face_landmarks[0]
                    entry["eye_contact_score"]=estimate_eye_contact(lm,w,h)
                    pitch,yaw=estimate_head_pose(lm); entry["head_pitch"]=pitch; entry["head_yaw"]=yaw
                    eng=estimate_engagement(lm); entry["expression"]=classify_expression(eng)
                    curr_nose_y=lm.landmark[NOSE_TIP].y
                    if prev_nose_y is not None and detect_nod(prev_nose_y,curr_nose_y): nod_events.append(round(ts,2))
                    prev_nose_y=curr_nose_y
                else:
                    entry.update({"eye_contact_score":0.0,"head_pitch":"No face","head_yaw":"No face","expression":"No face detected"})
                    prev_nose_y=None
                frames_data.append(entry); sampled+=1
                if progress_cb: progress_cb(min(frame_idx/max(total,1),0.95))
            frame_idx+=1
    cap.release()
    detected=[f for f in frames_data if f["face_found"]]
    detection_rate=len(detected)/max(len(frames_data),1)
    eye_scores=[f["eye_contact_score"] for f in detected]
    avg_eye=float(np.mean(eye_scores)) if eye_scores else 0
    eye_pct=float(np.mean([s>0.45 for s in eye_scores])) if eye_scores else 0
    expr_counts={}
    for f in detected: expr_counts[f["expression"]]=expr_counts.get(f["expression"],0)+1
    head_yaws=[f["head_yaw"] for f in detected]
    forward_pct=sum(1 for y in head_yaws if "forward" in y.lower())/max(len(head_yaws),1)
    gaze_events=[f["timestamp"] for f in detected if f["eye_contact_score"]<0.3]
    gaze_windows=[]
    if gaze_events:
        start=prev=gaze_events[0]
        for t in gaze_events[1:]:
            if t-prev>3: gaze_windows.append((round(start),round(prev))); start=t
            prev=t
        gaze_windows.append((round(start),round(prev)))
    nod_windows=[]
    if nod_events:
        start=prev=nod_events[0]
        for t in nod_events[1:]:
            if t-prev>2: nod_windows.append(round(start)); start=t
            prev=t
        nod_windows.append(round(start))
    return {"duration_sec":round(duration,1),"total_frames":total,"sampled_frames":sampled,
            "detection_rate":round(detection_rate*100,1),"avg_eye_contact":round(avg_eye,3),
            "eye_contact_pct":round(eye_pct*100,1),"forward_pct":round(forward_pct*100,1),
            "expr_counts":expr_counts,"gaze_away_windows":gaze_windows,
            "nod_timestamps":nod_windows[:20],"frames_data":frames_data,
            "fps":round(fps,1),"resolution":f"{w}x{h}"}

# ─────────────────────────────────────────────
# AUDIO EXTRACTION
# ─────────────────────────────────────────────
def extract_audio(video_path, output_dir):
    audio_path=os.path.join(output_dir,"audio.mp3")
    result=subprocess.run(["ffmpeg","-i",video_path,"-vn","-ar","16000","-ac","1","-b:a","64k","-y",audio_path],
                          capture_output=True,text=True,timeout=120)
    if result.returncode!=0: raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")
    if not os.path.exists(audio_path) or os.path.getsize(audio_path)<1000: raise RuntimeError("Audio extraction produced empty file")
    return audio_path

# ─────────────────────────────────────────────
# UPGRADED: GROQ WHISPER + PYANNOTE DIARIZATION
# ─────────────────────────────────────────────
# Groq Whisper  → WHAT was said + timestamps (cloud, fast)
# PyAnnote 3.1  → WHO said it + timestamps  (local, Mac M1 MPS)
# Merge         → Match using timestamp overlap
# ─────────────────────────────────────────────

def _merge_whisper_pyannote(whisper_segments, pyannote_segments):
    """
    Bridge between Whisper and PyAnnote using timestamps.

    For each Whisper text segment:
      Find the PyAnnote speaker with maximum timestamp overlap
      Assign that speaker label to the text

    Whisper  → WHAT was said (text + timestamps)
    PyAnnote → WHO said it   (speaker + timestamps)
    Timestamps → the bridge that connects both
    """
    utterances = []
    for seg in whisper_segments:
        seg_start = seg.get("start", 0)
        seg_end   = seg.get("end",   0)
        seg_text  = seg.get("text",  "").strip()

        best_speaker = "SPEAKER_00"
        best_overlap = -1

        for ps in pyannote_segments:
            # Calculate overlap between Whisper segment and PyAnnote segment
            overlap = min(seg_end, ps["end"]) - max(seg_start, ps["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = ps["speaker"]

        utterances.append({
            "speaker"   : best_speaker,
            "start"     : int(seg_start * 1000),
            "end"       : int(seg_end   * 1000),
            "text"      : seg_text,
            "confidence": seg.get("avg_logprob", 0),
        })
    return utterances


def _heuristic_diarization(segments, full_text):
    """
    Fallback heuristic speaker detection.
    Used ONLY when PyAnnote is unavailable.
    Less accurate (~60%) vs PyAnnote (~94%).
    """
    utterances = []; current_speaker = "A"
    for i, seg in enumerate(segments):
        text     = seg.get("text", "").strip()
        start_ms = int(seg.get("start", 0) * 1000)
        end_ms   = int(seg.get("end",   0) * 1000)
        if i > 0:
            prev      = segments[i - 1]
            prev_text = prev.get("text", "").strip()
            gap       = seg.get("start", 0) - prev.get("end", 0)
            if prev_text.endswith("?") or gap > 0.8:
                current_speaker = "B" if current_speaker == "A" else "A"
        utterances.append({
            "speaker"   : current_speaker,
            "start"     : start_ms,
            "end"       : end_ms,
            "text"      : text,
            "confidence": seg.get("avg_logprob", 0),
        })
    return {"utterances": utterances, "text": full_text,
            "diarization_method": "heuristic"}


def transcribe_with_diarization(audio_path):
    """
    UPGRADED: Groq Whisper + PyAnnote diarization pipeline

    Step 1 → Groq Whisper API (cloud):
              Sends audio to Groq servers
              Returns: text + timestamps per segment
              Does NOT return speaker labels

    Step 2 → PyAnnote 3.1 (local, Mac M1 MPS):
              Reads raw audio file directly
              Analyzes voice characteristics
              Returns: speaker labels + timestamps
              Does NOT return any text

    Step 3 → Merge (your code):
              Uses timestamps as bridge
              Combines WHAT (Whisper) + WHO (PyAnnote)
              Returns: labeled transcript

    Falls back to heuristic if PyAnnote unavailable
    """
    groq_key = st.session_state.get("groq_key_store", "")
    hf_token = st.session_state.get("hf_token_store", "")
    device   = get_device()

    # ── STEP 1: Groq Whisper — WHAT was said ────────────────────
    headers = {"Authorization": f"Bearer {groq_key}"}
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
        data  = {
            "model"          : "whisper-large-v3",
            "response_format": "verbose_json",
            "language"       : "en",
            "temperature"    : "0"
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers, files=files, data=data, timeout=120
        )

    if resp.status_code == 401: raise ValueError("Invalid Groq API key")
    if resp.status_code != 200:
        raise ValueError(f"Whisper failed {resp.status_code}: {resp.text[:200]}")

    whisper_result   = resp.json()
    whisper_segments = whisper_result.get("segments", [])
    full_text        = whisper_result.get("text", "")

    # ── STEP 2: PyAnnote — WHO said it ──────────────────────────
    if not hf_token:
        st.info("💡 Add HuggingFace token in sidebar for accurate speaker separation (~94% vs ~60%)")
        return _heuristic_diarization(whisper_segments, full_text)

    try:
        from pyannote.audio import Pipeline

        # Load PyAnnote pipeline
        # First run: downloads ~1GB model to ~/.cache/huggingface
        # Subsequent runs: loads instantly from cache
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        # Move to Mac M1 MPS for acceleration
        # MPS = Metal Performance Shaders (Apple Silicon GPU)
        if device == "mps":
            pipeline = pipeline.to(torch.device("mps"))
        elif device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))
        # CPU: no .to() needed

        # Run diarization on RAW AUDIO — not the transcript
        # PyAnnote analyzes voice characteristics / fingerprints
        diarization = pipeline(
            audio_path,
            min_speakers=2,  # doctor + patient
            max_speakers=2   # exactly 2 speakers
        )

        # Convert PyAnnote output to list of dicts
        # itertracks gives: turn (start/end), track_id, speaker_label
        pyannote_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            pyannote_segments.append({
                "start"  : turn.start,   # seconds (float)
                "end"    : turn.end,     # seconds (float)
                "speaker": speaker       # "SPEAKER_00" or "SPEAKER_01"
            })

        # ── STEP 3: Merge Whisper + PyAnnote ────────────────────
        # Timestamps are the bridge between the two systems
        utterances = _merge_whisper_pyannote(
            whisper_segments,
            pyannote_segments
        )

        return {
            "utterances"         : utterances,
            "text"               : full_text,
            "diarization_method" : "pyannote",
            "device_used"        : device.upper(),
        }

    except ImportError:
        st.warning("⚠️ PyAnnote not installed. Run: pip install pyannote.audio torch")
        return _heuristic_diarization(whisper_segments, full_text)

    except Exception as e:
        st.warning(f"⚠️ PyAnnote failed ({str(e)[:120]}) — using heuristic fallback")
        return _heuristic_diarization(whisper_segments, full_text)


def identify_speakers(utterances):
    """
    Identify which speaker label is doctor vs patient.
    Heuristic: doctor asks more questions.
    Works for both PyAnnote labels (SPEAKER_00/01)
    and heuristic labels (A/B).
    """
    if not utterances: return {"doctor":"A","patient":"B"}
    qc={}
    for utt in utterances: qc[utt["speaker"]]=qc.get(utt["speaker"],0)+utt["text"].count("?")
    doctor_label=max(qc,key=qc.get) if qc else list(qc.keys())[0]
    all_speakers=sorted({u["speaker"] for u in utterances})
    patient_label=next((s for s in all_speakers if s!=doctor_label),"SPEAKER_01")
    return {"doctor":doctor_label,"patient":patient_label}


def process_diarized_transcript(raw_transcript):
    processed=[]
    for utt in raw_transcript.get("utterances",[]):
        processed.append({
            "speaker"   : utt.get("speaker","?"),
            "start_s"   : round(utt.get("start",0)/1000,1),
            "end_s"     : round(utt.get("end",0)/1000,1),
            "text"      : utt.get("text","").strip(),
            "confidence": round(utt.get("confidence",0),2),
            "words"     : len(utt.get("text","").split())
        })
    return processed


def analyze_per_speaker(utterances, speaker_map):
    dl=speaker_map["doctor"]; pl=speaker_map["patient"]
    du=[u for u in utterances if u["speaker"]==dl]
    pu=[u for u in utterances if u["speaker"]==pl]
    def stats(utts):
        if not utts: return {"word_count":0,"wpm":0,"filler_count":0,"turn_count":0,"avg_turn_duration":0}
        tw=sum(u["words"] for u in utts); td=sum(u["end_s"]-u["start_s"] for u in utts)
        fc=sum(sum(1 for fw in FILLER_WORDS if fw in u["text"].lower()) for u in utts)
        return {"word_count":tw,"wpm":round(tw/max(td,1)*60,1),"filler_count":fc,
                "turn_count":len(utts),"avg_turn_duration":round(td/len(utts),1)}
    latencies=[]
    for i,utt in enumerate(utterances[:-1]):
        nxt=utterances[i+1]
        if utt["speaker"]==dl and nxt["speaker"]==pl:
            gap=round(nxt["start_s"]-utt["end_s"],2)
            if 0<=gap<15: latencies.append({
                "after_doctor_end":utt["end_s"],
                "patient_starts"  :nxt["start_s"],
                "latency_s"       :gap,
                "doctor_said"     :utt["text"][:60]
            })
    avg_lat=round(np.mean([l["latency_s"] for l in latencies]),2) if latencies else 0
    tt=len(utterances); dt=len(du); pt=len(pu)
    return {"doctor":{**stats(du),"label":dl},"patient":{**stats(pu),"label":pl},
            "response_latencies":latencies[:10],"avg_response_latency":avg_lat,
            "total_turns":tt,"doctor_turn_pct":round(dt/max(tt,1)*100,1),
            "patient_turn_pct":round(pt/max(tt,1)*100,1)}


def align_speech_with_face(utterances, visual_stats, speaker_map):
    frames=visual_stats.get("frames_data",[]); aligned=[]
    for utt in utterances[:20]:
        s=utt["start_s"]; e=utt["end_s"]
        m=[f for f in frames if f.get("face_found") and s<=f["timestamp"]<=e]
        if m:
            eye=round(np.mean([f["eye_contact_score"] for f in m]),2)
            exprs=[f["expression"] for f in m]; dom_expr=max(set(exprs),key=exprs.count)
            yaws=[f["head_yaw"] for f in m]; dom_yaw=max(set(yaws),key=yaws.count)
        else: eye=None; dom_expr="Unknown"; dom_yaw="Unknown"
        aligned.append({"time":f"{s}s-{e}s","speaker":utt["speaker"],
                        "role":"DOCTOR" if utt["speaker"]==speaker_map["doctor"] else "PATIENT",
                        "speech":utt["text"][:120],"eye_contact":eye,
                        "expression":dom_expr,"head_pose":dom_yaw})
    return aligned

# ─────────────────────────────────────────────
# YOUTUBE DOWNLOAD
# ─────────────────────────────────────────────
def download_youtube(url, output_dir):
    import yt_dlp
    ydl_opts={"format":"bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]",
              "outtmpl":os.path.join(output_dir,"%(id)s.%(ext)s"),"quiet":True,"no_warnings":True,"merge_output_format":"mp4"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(url,download=True)
        for fn in Path(output_dir).iterdir():
            if fn.suffix in [".mp4",".mkv",".webm"]:
                return str(fn),info.get("title","Unknown"),round(info.get("duration",0),0)
    raise FileNotFoundError("Download failed")

# ─────────────────────────────────────────────
# GROQ LLM — LLAMA 3.3 70B
# ─────────────────────────────────────────────
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"

def build_prompt(visual_stats, speaker_stats, utterances, speaker_map, aligned, topsis):
    gaze_str=", ".join(f"{s}s-{e}s" for s,e in visual_stats["gaze_away_windows"][:6]) or "None"
    nod_str =", ".join(f"{t}s" for t in visual_stats.get("nod_timestamps",[])[:8]) or "None"
    expr_str="\n".join(f"  - {expr}: {cnt} frames ({round(cnt/max(sum(visual_stats['expr_counts'].values()),1)*100)}%)"
                       for expr,cnt in sorted(visual_stats["expr_counts"].items(),key=lambda x:-x[1]))
    lat_str ="\n".join(f"  [{l['after_doctor_end']}s] Doctor: \"{l['doctor_said']}...\" → Patient after {l['latency_s']}s"
                       for l in speaker_stats.get("response_latencies",[])[:5]) or "No latency data"
    convo_str="\n".join(f"  {'DOCTOR' if u['speaker']==speaker_map['doctor'] else 'PATIENT'} ({u['start_s']}s): \"{u['text'][:80]}\""
                        for u in utterances[:16])
    aligned_str="\n".join(f"  [{ev['time']}] {ev['role']}: \"{ev['speech'][:70]}\"\n    Eye: {ev['eye_contact']} | Expr: {ev['expression']} | Head: {ev['head_pose']}"
                          for ev in aligned[:10])
    bd_str="\n".join(f"  {v['icon']} {v['label']} ({v['weight_pct']}): {v['normalized']}/100 [raw: {v['raw_value']}]"
                     for k,v in topsis["breakdown"].items())
    doc=speaker_stats["doctor"]; pat=speaker_stats["patient"]
    return f"""You are an expert clinical communication coach analysing a doctor-patient interaction.
The overall score is calculated algorithmically via TOPSIS + Calgary-Cambridge framework.

━━━ TOPSIS SCORE ━━━
Overall: {topsis['topsis_score']}/100 — {topsis['grade']}
d+: {topsis['d_plus']} | d-: {topsis['d_minus']}
{bd_str}

━━━ VISUAL ━━━
Duration: {visual_stats['duration_sec']}s | Detection: {visual_stats['detection_rate']}%
Eye Contact avg: {visual_stats['avg_eye_contact']:.2f}/1.0 | Strong gaze: {visual_stats['eye_contact_pct']}%
Gaze-Away: {gaze_str} | Nodding: {nod_str}
{expr_str}

━━━ DOCTOR ━━━
Words: {doc['word_count']} | WPM: {doc['wpm']} | Fillers: {doc['filler_count']} | Turns: {doc['turn_count']}

━━━ PATIENT ━━━
Words: {pat['word_count']} | WPM: {pat['wpm']} | Turns: {pat['turn_count']}

━━━ TURN-TAKING ━━━
Doctor: {speaker_stats['doctor_turn_pct']}% | Patient: {speaker_stats['patient_turn_pct']}%
Avg patient response latency: {speaker_stats['avg_response_latency']}s
{lat_str}

━━━ CONVERSATION ━━━
{convo_str}

━━━ ALIGNMENT ━━━
{aligned_str}

━━━ INSTRUCTIONS ━━━
Use {topsis['topsis_score']} as overall_score exactly. Coach the DOCTOR only.
Return ONLY valid JSON, no markdown fences. Keep all strings under 50 words:

{{
  "topsis_explanation": "<2 sentences explaining the score>",
  "summary": "<3 sentences coaching the doctor>",
  "eye_contact_feedback": {{"score": {round(topsis['breakdown']['eye_contact']['normalized'])}, "assessment": "<clinical impact>", "recommendations": ["<tip>","<tip>","<tip>"]}},
  "doctor_speech_feedback": {{"score": {round(topsis['breakdown']['speech_clarity']['normalized'])}, "assessment": "<WPM and filler impact>", "recommendations": ["<tip>","<tip>","<tip>"]}},
  "listening_feedback": {{"score": {round(topsis['breakdown']['turn_balance']['normalized'])}, "assessment": "<turn dominance impact>", "recommendations": ["<tip>","<tip>"]}},
  "body_language_feedback": {{"score": {round(topsis['breakdown']['body_language']['normalized'])}, "assessment": "<nodding and head pose>", "recommendations": ["<tip>","<tip>"]}},
  "patient_impact_analysis": {{"score": {round(topsis['breakdown']['response_latency']['normalized'])}, "assessment": "<patient comfort>", "key_moments": ["<moment>"], "recommendations": ["<tip>","<tip>"]}},
  "behavioural_insights": ["<doctor behaviour + patient effect>","<insight 2>","<insight 3>"],
  "priority_actions": [{{"rank":1,"action":"<change>","rationale":"<evidence>"}},{{"rank":2,"action":"<change>","rationale":"<why>"}},{{"rank":3,"action":"<change>","rationale":"<why>"}}],
  "strengths": ["<strength>","<strength>"],
  "coaching_tips": ["<tip>","<tip>","<tip>"]
}}""".strip()


def get_llm_feedback(visual_stats, speaker_stats, utterances, speaker_map, aligned, topsis, groq_key):
    prompt=build_prompt(visual_stats,speaker_stats,utterances,speaker_map,aligned,topsis)
    headers={"Authorization":f"Bearer {groq_key.strip()}","Content-Type":"application/json"}
    payload={"model":GROQ_MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":2800,"temperature":0}
    last_err=None
    for attempt in range(3):
        try:
            resp=requests.post(GROQ_CHAT_URL,headers=headers,json=payload,timeout=60)
            if resp.status_code==401: raise ValueError("Invalid Groq API key")
            if resp.status_code==429: time.sleep(5*(attempt+1)); continue
            resp.raise_for_status()
            raw=resp.json()["choices"][0]["message"]["content"].strip()
            if "```" in raw:
                for part in raw.split("```"):
                    part=part.strip()
                    if part.startswith("json"): part=part[4:].strip()
                    if part.startswith("{"): raw=part; break
            raw=raw.strip()
            if not raw.endswith("}"):
                depth=0; last_valid=0
                for idx,ch in enumerate(raw):
                    if ch=="{": depth+=1
                    elif ch=="}":
                        depth-=1
                        if depth==0: last_valid=idx
                raw=raw[:last_valid+1] if last_valid>0 else raw[:raw.rfind('","')+3]+"}"
            return json.loads(raw)
        except (requests.exceptions.ConnectionError,requests.exceptions.Timeout) as e:
            last_err=e; time.sleep(3*(attempt+1))
        except json.JSONDecodeError as e: raise ValueError(f"Could not parse Groq JSON: {e}")
        except ValueError: raise
    raise ConnectionError(f"Could not reach Groq after 3 attempts: {last_err}")

# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────
def score_class(s):
    return "score-excellent" if s>=80 else "score-good" if s>=60 else "score-fair" if s>=40 else "score-poor"

def render_metric(label, value, sub="", color_class="score-good"):
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value {color_class}">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

def render_progress(label, pct, color="#3182ce"):
    st.markdown(f'<div style="margin-bottom:.5rem"><div style="display:flex;justify-content:space-between;font-size:.85rem;color:#4a5568;margin-bottom:.25rem"><span>{label}</span><span><b>{pct}%</b></span></div><div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct}%;background:{color}"></div></div></div>', unsafe_allow_html=True)

def render_feedback_block(title, icon, data):
    score=data.get("score",0)
    with st.expander(f"{icon} {title} — Score: {score}/100", expanded=True):
        st.markdown(f'<div style="background:#f7fafc;border-radius:8px;padding:1rem;margin-bottom:1rem;font-size:.9rem;color:#4a5568;line-height:1.6">{data.get("assessment","No data.")}</div>', unsafe_allow_html=True)
        for r in data.get("recommendations",[]): st.markdown(f"- {r}")

def pinned_block(feedback, fb_key, topsis_key, topsis):
    block = dict(feedback.get(fb_key, {}))
    block["score"] = round(topsis["breakdown"][topsis_key]["normalized"])
    return block

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ API Keys")

    groq_key=st.text_input(
        "Groq API Key", type="password",
        help="Free at console.groq.com — for Whisper transcription + Llama feedback",
        placeholder="gsk_..."
    )
    if groq_key: st.session_state["groq_key_store"]=groq_key
    if groq_key and not groq_key.strip().startswith("gsk_"):
        st.warning("⚠️ Groq keys start with **gsk_**")

    st.divider()

    # ── PYANNOTE / HUGGINGFACE TOKEN ──────────────────────────
    st.markdown("### 🤗 HuggingFace Token")
    hf_token=st.text_input(
        "HuggingFace Token", type="password",
        help="Free at huggingface.co — required for PyAnnote speaker diarization",
        placeholder="hf_..."
    )
    if hf_token: st.session_state["hf_token_store"]=hf_token

    # Show diarization mode status
    device = get_device()
    if hf_token:
        st.success(f"🎙️ PyAnnote active · ~94% accuracy")
        st.caption(f"Device: {device.upper()} · Mac M1 MPS accelerated")
    else:
        st.warning("⚠️ Heuristic diarization · ~60% accuracy")
        st.caption("Add HF token for PyAnnote speaker separation")

    st.divider()
    st.markdown("### 🖥️ View Mode")
    view_mode=st.radio("Select mode",["👨‍⚕️ Quick Feedback","🔬 Research Mode"],index=0,
                       help="Quick = simple & actionable | Research = full analysis")
    is_quick=view_mode=="👨‍⚕️ Quick Feedback"

    st.divider()
    st.markdown("""
    <div style="background:#f0fff4;border:1px solid #9ae6b4;border-radius:8px;
                padding:.8rem 1rem;font-size:.82rem;color:#276749">
        ⚙️ <b>Pipeline:</b><br>
        • Groq Whisper → transcription<br>
        • PyAnnote → speaker diarization<br>
        • Every 4th frame sampled<br>
        • TOPSIS + Calgary-Cambridge scoring
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    ### 📋 Visual Detection
    - 👁️ Eye contact & gaze
    - 😊 Facial expressions
    - 🙆 Head pose + nodding
    - 📉 Gaze-away timestamps

    ### 🎙️ Audio Detection
    - 👨‍⚕️ Doctor speech separated
    - 🧑 Patient speech separated
    - ⏱️ Response latency per turn
    - 🐌 Filler words per speaker
    - 💬 WPM per speaker

    ### 🤖 AI Feedback
    - TOPSIS + Calgary-Cambridge score
    - Scores pinned to algorithm
    - Doctor-only coaching
    - Cross-modal insights

    ### 🔑 Free Keys
    - **console.groq.com** (Groq)
    - **huggingface.co** (HuggingFace)
    """)
    st.divider()
    st.caption("v3 — Groq Whisper + PyAnnote + TOPSIS")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Doctor-Patient Communication Analyzer</h1>
    <p>v3 · Groq Whisper + PyAnnote (Mac M1) + Groq Llama 3.3 70B · TOPSIS Calgary-Cambridge Scoring</p>
</div>""", unsafe_allow_html=True)

col_url,col_btn=st.columns([5,1])
with col_url:
    youtube_url=st.text_input("YouTube URL",placeholder="https://www.youtube.com/watch?v=...",label_visibility="collapsed")
with col_btn:
    analyze_btn=st.button("▶ Analyze",use_container_width=True)

st.markdown('<div class="info-box">💡 <b>Best results:</b> Use a video with clear audio of both doctor and patient. Try: <b>"OSCE clinical communication"</b> or <b>"doctor patient consultation"</b></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
if analyze_btn and youtube_url:
    if not groq_key: st.error("⚠️ Enter your Groq API key in the sidebar."); st.stop()
    tmp_dir=tempfile.mkdtemp()

    # STEP 1 — Download
    with st.status("📥 Downloading video…",expanded=True) as status:
        try:
            video_path,title,vid_duration=download_youtube(youtube_url,tmp_dir)
            status.update(label=f"✅ Downloaded: **{title}** ({int(vid_duration)}s)",state="complete")
        except Exception as e: st.error(f"Download failed: {e}"); st.stop()

    # STEP 2 — MediaPipe visual analysis
    prog_bar=st.progress(0,text="🔍 Analyzing frames with MediaPipe…")
    def update_prog(v): prog_bar.progress(v,text=f"🔍 Analyzing frames… {int(v*100)}%")
    with st.spinner(""):
        visual_stats=analyze_video(video_path,sample_rate=4,max_frames=500,progress_cb=update_prog)
    prog_bar.progress(1.0,text="✅ Visual analysis complete"); time.sleep(0.2); prog_bar.empty()

    # STEP 3 — Groq Whisper + PyAnnote diarization
    utterances=[]; speaker_map={"doctor":"SPEAKER_00","patient":"SPEAKER_01"}
    speaker_stats={}; audio_path=None; diarization_method="unknown"

    with st.status("🎙️ Groq Whisper transcription + PyAnnote diarization…",expanded=True) as audio_status:
        try:
            audio_path    = extract_audio(video_path, tmp_dir)
            raw           = transcribe_with_diarization(audio_path)
            diarization_method = raw.get("diarization_method", "unknown")
            utterances    = process_diarized_transcript(raw)
            speaker_map   = identify_speakers(utterances)
            speaker_stats = analyze_per_speaker(utterances, speaker_map)
            doc = speaker_stats["doctor"]; pat = speaker_stats["patient"]

            method_label = (
                f"PyAnnote (~94% accuracy) · {raw.get('device_used','')}"
                if diarization_method == "pyannote"
                else "Heuristic (~60% accuracy) — add HF token for better results"
            )
            audio_status.update(
                label=(
                    f"✅ Doctor: {doc['word_count']} words, {doc['wpm']} WPM, "
                    f"{doc['filler_count']} fillers | "
                    f"Patient: {pat['word_count']} words, {pat['wpm']} WPM | "
                    f"Avg latency: {speaker_stats['avg_response_latency']}s | "
                    f"Diarization: {method_label}"
                ),
                state="complete"
            )
        except Exception as e:
            audio_status.update(label=f"⚠️ Diarization failed: {e}",state="error")
            speaker_stats={
                "doctor" :{"word_count":0,"wpm":0,"filler_count":None,"turn_count":0,"avg_turn_duration":0,"label":"SPEAKER_00"},
                "patient":{"word_count":0,"wpm":0,"filler_count":0,   "turn_count":0,"avg_turn_duration":0,"label":"SPEAKER_01"},
                "response_latencies":[],"avg_response_latency":None,
                "total_turns":0,"doctor_turn_pct":None,"patient_turn_pct":None,
                "_diarization_failed":True
            }

    # STEP 4 — TOPSIS scoring
    with st.spinner("📐 Calculating TOPSIS + Calgary-Cambridge score…"):
        topsis=calculate_topsis_score(visual_stats,speaker_stats)

    # STEP 5 — Groq Llama feedback
    aligned=align_speech_with_face(utterances,visual_stats,speaker_map)
    feedback=None
    with st.spinner("🤖 Generating AI feedback via Groq Llama 3.3…"):
        try: feedback=get_llm_feedback(visual_stats,speaker_stats,utterances,speaker_map,aligned,topsis,groq_key)
        except Exception as e: st.error(f"❌ AI feedback error: {e}")

    # Force TOPSIS score — LLM cannot change it
    if feedback:
        feedback["overall_score"] = topsis["topsis_score"]
        feedback["overall_grade"] = topsis["grade"]

    st.success(f"✅ Full analysis complete for **{title}**"); st.divider()
    doc=speaker_stats.get("doctor",{}); pat=speaker_stats.get("patient",{})

    # ══════════════════════════════════════════
    # QUICK MODE
    # ══════════════════════════════════════════
    if is_quick:
        st.markdown('<span class="mode-badge mode-quick">👨‍⚕️ Quick Feedback</span>', unsafe_allow_html=True)
        score=topsis["topsis_score"]; grade=topsis["grade"]
        score_color=("#38a169" if score>=80 else "#3182ce" if score>=65 else "#d69e2e" if score>=45 else "#e53e3e")
        st.markdown(f'<div class="quick-score-card"><div class="big-score" style="color:{score_color}">{score}/100</div><div class="grade" style="color:{score_color}">{grade}</div><div style="font-size:.85rem;color:#718096;margin-top:.5rem">Overall Communication Score · Calgary-Cambridge Framework</div></div>', unsafe_allow_html=True)

        st.markdown("#### How did the doctor do?")
        for key,val in topsis["breakdown"].items():
            n=val["normalized"]; col=("#38a169" if n>=70 else "#d69e2e" if n>=40 else "#e53e3e")
            st.markdown(f'<div style="background:white;border-radius:10px;padding:.8rem 1.2rem;border:1px solid #e2e8f0;margin-bottom:.5rem;display:flex;align-items:center;justify-content:space-between"><div><span style="font-size:1.1rem">{val["icon"]}</span><span style="font-weight:600;color:#2d3748;margin-left:.5rem">{val["label"]}</span><span style="font-size:.78rem;color:#718096;margin-left:.5rem">{val["dimension"]}</span></div><div style="font-size:1.2rem;font-weight:800;color:{col}">{val["status_icon"]} {round(n)}/100</div></div>', unsafe_allow_html=True)

        st.markdown("#### Performance Radar")
        st.plotly_chart(radar_chart(topsis),use_container_width=True)

        if feedback and feedback.get("summary"): st.info(f"📋 **Summary:** {feedback['summary']}")

        if feedback and feedback.get("priority_actions"):
            st.markdown("#### 🎯 Top 3 Actions")
            cols=st.columns(3)
            for action,col in zip(feedback["priority_actions"][:3],cols):
                rank=action.get("rank",1); color=["#e53e3e","#d69e2e","#3182ce"][min(rank-1,2)]
                with col:
                    st.markdown(f'<div style="background:white;border-left:4px solid {color};border-radius:8px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,.06);height:100%"><div style="font-size:.75rem;font-weight:700;color:{color};text-transform:uppercase">#{rank} Priority</div><div style="font-size:.9rem;font-weight:600;color:#2d3748;margin:.3rem 0">{action.get("action","")}</div><div style="font-size:.8rem;color:#718096">{action.get("rationale","")}</div></div>', unsafe_allow_html=True)

        if feedback and feedback.get("strengths"):
            st.markdown("#### ✅ Strengths")
            for s in feedback["strengths"]: st.markdown(f"<span class='tag tag-green'>✓ {s}</span>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # RESEARCH MODE
    # ══════════════════════════════════════════
    else:
        st.markdown('<span class="mode-badge mode-research">🔬 Research Mode</span>', unsafe_allow_html=True)

        st.markdown(f'<div class="topsis-box"><b>📐 TOPSIS Score (Calgary-Cambridge Framework)</b><br><br><b>Overall: {topsis["topsis_score"]}/100 — {topsis["grade"]}</b><br>Distance from ideal doctor (d+): <b>{topsis["d_plus"]}</b> &nbsp;|&nbsp; Distance from worst doctor (d-): <b>{topsis["d_minus"]}</b><br><br>{feedback.get("topsis_explanation","") if feedback else ""}</div>', unsafe_allow_html=True)

        c1,c2,c3,c4,c5,c6,c7=st.columns(7)
        with c1: render_metric("Overall",f"{topsis['topsis_score']}/100",topsis["grade"],score_class(topsis["topsis_score"]))
        with c2: render_metric("Eye Contact",f"{visual_stats['eye_contact_pct']}%","strong gaze",score_class(int(visual_stats['eye_contact_pct'])))
        with c3: render_metric("Doctor WPM",str(doc.get("wpm",0)),"speech rate","score-good" if 110<=doc.get("wpm",0)<=160 else "score-fair")
        with c4: render_metric("Patient WPM",str(pat.get("wpm",0)),"speech rate","score-good" if pat.get("wpm",0)>50 else "score-fair")
        with c5: render_metric("Dr Fillers",str(doc.get("filler_count",0)),"filler words","score-good" if doc.get("filler_count",0)<5 else "score-fair" if doc.get("filler_count",0)<15 else "score-poor")
        with c6:
            lat=speaker_stats.get("avg_response_latency",0)
            render_metric("Avg Response",f"{lat}s","patient latency","score-good" if lat<2 else "score-fair" if lat<4 else "score-poor")
        with c7: render_metric("Turn Balance",f"{speaker_stats.get('doctor_turn_pct',0)}%","doctor turns","score-good" if 40<=speaker_stats.get('doctor_turn_pct',0)<=60 else "score-fair")

        st.markdown("<br>", unsafe_allow_html=True)
        rc,bc=st.columns(2)
        with rc:
            st.markdown("#### 📊 Calgary-Cambridge Radar")
            st.plotly_chart(radar_chart(topsis),use_container_width=True)
        with bc:
            st.markdown("#### 📐 TOPSIS Dimension Breakdown")
            for key,val in topsis["breakdown"].items():
                n=val["normalized"]; col=("#38a169" if n>=70 else "#d69e2e" if n>=40 else "#e53e3e")
                st.markdown(f'<div style="background:white;border-radius:8px;padding:.7rem 1rem;border:1px solid #e2e8f0;margin-bottom:.5rem"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem"><span style="font-weight:600;font-size:.88rem;color:#2d3748">{val["icon"]} {val["label"]} ({val["weight_pct"]})</span><span style="font-weight:800;color:{col}">{round(n)}/100</span></div><div style="background:#edf2f7;border-radius:999px;height:6px;overflow:hidden"><div style="width:{n}%;height:100%;background:{col};border-radius:999px"></div></div><div style="font-size:.74rem;color:#718096;margin-top:.3rem">{val["dimension"]} · Raw: {val["raw_value"]} · Weighted: {val["weighted"]}</div></div>', unsafe_allow_html=True)

        st.divider()

        if feedback and feedback.get("summary"):
            st.markdown(f'<div class="feedback-section"><h3>📋 Executive Summary</h3><p style="color:#4a5568;line-height:1.7;font-size:.95rem">{feedback["summary"]}</p></div>', unsafe_allow_html=True)

        if utterances:
            st.markdown("### 💬 Conversation Overview")
            st.caption(f"Showing first 6 turns · 🔵 Doctor · 🟢 Patient · Diarization: {diarization_method.upper()}")
            for i,utt in enumerate(utterances[:6]):
                is_doctor=utt["speaker"]==speaker_map["doctor"]
                role="👨‍⚕️ Doctor" if is_doctor else "🧑 Patient"
                avatar="👨‍⚕️" if is_doctor else "🧑"
                ts=f"{utt.get('start_s',0)}s"; latency=""
                if not is_doctor and i>0:
                    prev=utterances[i-1]
                    if prev["speaker"]==speaker_map["doctor"]:
                        gap=round(utt.get("start_s",0)-prev.get("end_s",0),1)
                        if 0<gap<15: latency=f"  ⏱ *{gap}s to respond*"
                with st.chat_message(name=role,avatar=avatar):
                    st.markdown(f"**{ts}**{latency}"); st.write(utt["text"])

            if len(utterances)>6:
                with st.expander(f"📄 Show full conversation ({len(utterances)} turns total)"):
                    for i,utt in enumerate(utterances):
                        is_doctor=utt["speaker"]==speaker_map["doctor"]
                        color="#2b6cb0" if is_doctor else "#276749"; bg="#ebf8ff" if is_doctor else "#f0fff4"
                        role="👨‍⚕️ Doctor" if is_doctor else "🧑 Patient"; latency=""
                        if not is_doctor and i>0:
                            prev=utterances[i-1]
                            if prev["speaker"]==speaker_map["doctor"]:
                                gap=round(utt["start_s"]-prev["end_s"],1)
                                if 0<=gap<15: latency=f" · ⏱ {gap}s"
                        st.markdown(f'<div style="background:{bg};border-radius:8px;padding:.5rem .8rem;margin-bottom:.3rem;border-left:3px solid {color}"><span style="color:{color};font-weight:700;font-size:.75rem">{role} · {utt["start_s"]}s{latency}</span><br><span style="color:#2d3748;font-size:.85rem">{utt["text"]}</span></div>', unsafe_allow_html=True)

        st.divider()

        dc,pc=st.columns(2)
        with dc:
            st.markdown("### 👨‍⚕️ Doctor Speech Stats")
            dm1,dm2,dm3=st.columns(3)
            with dm1: st.metric("Words",doc.get("word_count",0))
            with dm2: st.metric("WPM",doc.get("wpm",0))
            with dm3: st.metric("Fillers",doc.get("filler_count",0))
            st.metric("Turn count",doc.get("turn_count",0),f"Avg {doc.get('avg_turn_duration',0)}s per turn")
        with pc:
            st.markdown("### 🧑 Patient Speech Stats")
            pm1,pm2,pm3=st.columns(3)
            with pm1: st.metric("Words",pat.get("word_count",0))
            with pm2: st.metric("WPM",pat.get("wpm",0))
            with pm3: st.metric("Fillers",pat.get("filler_count",0))
            st.metric("Turn count",pat.get("turn_count",0),f"Avg {pat.get('avg_turn_duration',0)}s per turn")

        if speaker_stats.get("response_latencies"):
            st.markdown("#### ⏱️ Patient Response Latency per Turn")
            for lat in speaker_stats["response_latencies"][:6]:
                color=("#38a169" if lat["latency_s"]<2 else "#d69e2e" if lat["latency_s"]<4 else "#e53e3e")
                st.markdown(f'<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:.7rem 1rem;margin-bottom:.4rem;font-size:.83rem"><div style="color:#3182ce;font-weight:600;font-size:.75rem">{lat["after_doctor_end"]}s</div><div style="color:#4a5568;margin:.2rem 0">Doctor: <i>"{lat["doctor_said"]}..."</i></div><div style="color:{color};font-weight:700">⏱ Patient responded in {lat["latency_s"]}s</div></div>', unsafe_allow_html=True)

        st.divider()

        st.markdown("### 👁️ Visual Analysis")
        v1,v2=st.columns(2)
        with v1:
            total_expr=sum(visual_stats["expr_counts"].values())
            color_map={"Neutral":"#4299e1","Attentive / Raised brow":"#38a169","Speaking / Reacting":"#d69e2e","Concerned / Furrowed":"#e53e3e","No face detected":"#a0aec0"}
            for expr,cnt in sorted(visual_stats["expr_counts"].items(),key=lambda x:-x[1]):
                pct=round(cnt/max(total_expr,1)*100); render_progress(expr,pct,color_map.get(expr,"#718096"))
        with v2:
            if visual_stats["gaze_away_windows"]:
                st.markdown("#### ⚠️ Gaze-Away Moments")
                for s,e in visual_stats["gaze_away_windows"][:5]:
                    st.markdown(f'<div class="timeline-item"><div class="timeline-time">⏱ {s}s – {e}s</div><div class="timeline-text">Reduced eye contact</div></div>', unsafe_allow_html=True)
            if visual_stats.get("nod_timestamps"):
                nod_str=" · ".join(f"{t}s" for t in visual_stats["nod_timestamps"][:8])
                st.markdown(f'<div class="info-box">🔄 Nodding at: {nod_str}</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown("### 🔬 Detailed AI Feedback")
        if feedback:
            insights=feedback.get("behavioural_insights",[])
            if insights:
                st.markdown("#### 🔍 Doctor Behavioural Insights")
                for ins in insights:
                    st.markdown(f'<div style="background:#faf5ff;border-left:4px solid #9f7aea;border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem;font-size:.88rem;color:#553c9a">💡 {ins}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fc1,fc2=st.columns(2)
            with fc1:
                render_feedback_block("Eye Contact",      "👁️", pinned_block(feedback,"eye_contact_feedback",   "eye_contact",      topsis))
                render_feedback_block("Doctor's Speech",  "🎙️", pinned_block(feedback,"doctor_speech_feedback", "speech_clarity",   topsis))
                render_feedback_block("Listening Skills", "👂", pinned_block(feedback,"listening_feedback",      "turn_balance",     topsis))
            with fc2:
                render_feedback_block("Body Language",    "🙆", pinned_block(feedback,"body_language_feedback",  "body_language",    topsis))
                render_feedback_block("Patient Impact",   "💙", pinned_block(feedback,"patient_impact_analysis", "response_latency", topsis))

            st.markdown("### 🎯 Priority Actions")
            if feedback.get("priority_actions"):
                p1,p2,p3=st.columns(3)
                for action,col in zip(feedback["priority_actions"],[p1,p2,p3]):
                    rank=action.get("rank",1); color=["#e53e3e","#d69e2e","#3182ce"][min(rank-1,2)]
                    with col:
                        st.markdown(f'<div style="background:white;border-left:4px solid {color};border-radius:8px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,.06)"><div style="font-size:.75rem;font-weight:700;color:{color};text-transform:uppercase">#{rank} Priority</div><div style="font-size:.9rem;font-weight:600;color:#2d3748;margin:.3rem 0">{action.get("action","")}</div><div style="font-size:.8rem;color:#718096">{action.get("rationale","")}</div></div>', unsafe_allow_html=True)

            if feedback.get("strengths"):
                st.markdown("#### ✅ Strengths")
                for s in feedback["strengths"]: st.markdown(f"<span class='tag tag-green'>✓ {s}</span>", unsafe_allow_html=True)

            tips=feedback.get("coaching_tips",[])
            if tips:
                st.markdown("### 💡 Coaching Tips for the Doctor")
                tip_icons=["🗣️","👂","🤝","💬","🌡️"]; cols=st.columns(len(tips))
                for i,(tip,col) in enumerate(zip(tips,cols)):
                    with col:
                        st.markdown(f'<div style="background:linear-gradient(135deg,#ebf8ff,#e6fffa);border-radius:10px;padding:1rem;text-align:center;border:1px solid #bee3f8;height:100%"><div style="font-size:1.5rem">{tip_icons[i%len(tip_icons)]}</div><div style="font-size:.85rem;color:#2c5282;margin-top:.5rem;line-height:1.5">{tip}</div></div>', unsafe_allow_html=True)

        with st.expander("🗃️ Raw Analysis Data (JSON)"):
            st.json({
                "algorithm"          : "TOPSIS + Calgary-Cambridge",
                "diarization_method" : diarization_method,
                "topsis_result"      : {k:v for k,v in topsis.items() if k!="breakdown"},
                "topsis_breakdown"   : {k:{"label":v["label"],"raw_value":v["raw_value"],
                                           "normalized":v["normalized"],"weighted":v["weighted"],
                                           "weight":v["weight_pct"],"dimension":v["dimension"]}
                                        for k,v in topsis["breakdown"].items()},
                "visual_stats"  : {k:v for k,v in visual_stats.items() if k!="frames_data"},
                "speaker_stats" : {k:v for k,v in speaker_stats.items() if k!="response_latencies"},
                "utterances"    : utterances[:20],
                "ai_feedback"   : feedback,
            })

    try:
        for f in [video_path,audio_path]:
            if f and os.path.exists(f): os.remove(f)
        os.rmdir(tmp_dir)
    except Exception: pass

elif analyze_btn and not youtube_url:
    st.warning("Please enter a YouTube URL to analyze.")

st.markdown('<div style="text-align:center;color:#a0aec0;font-size:.8rem;margin-top:3rem;padding:1rem;border-top:1px solid #e2e8f0">v3 · Groq Whisper + PyAnnote (Mac M1 MPS) + Groq Llama 3.3 70B · TOPSIS + Calgary-Cambridge Scoring</div>', unsafe_allow_html=True)
