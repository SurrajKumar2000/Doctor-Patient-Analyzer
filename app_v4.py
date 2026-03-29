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

.hesitation-card {
    background:#fff5f5; border:1px solid #fed7d7; border-radius:8px;
    padding:.7rem 1rem; margin-bottom:.4rem; font-size:.83rem;
}
.engagement-card {
    background:#f0fff4; border:1px solid #9ae6b4; border-radius:8px;
    padding:.7rem 1rem; margin-bottom:.4rem; font-size:.83rem;
}
.arc-card {
    background:#ebf8ff; border:1px solid #90cdf4; border-radius:8px;
    padding:1rem 1.2rem; margin-bottom:.8rem;
}

.stButton>button {
    background: linear-gradient(135deg,#2b6cb0,#2c7a7b); color: white;
    border: none; padding: .6rem 2rem; border-radius: 8px;
    font-weight: 600; font-size: .95rem; transition: all .2s; width: 100%;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 4px 12px rgba(43,108,176,.4); }

.info-box { background:#ebf8ff; border:1px solid #90cdf4; border-radius:8px;
            padding:1rem 1.2rem; color:#2c5282; font-size:.9rem; }
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
NOSE_TIP  = 1;   CHIN      = 152
LEFT_EAR  = 234; RIGHT_EAR = 454
LEFT_BROW  = [70,  63, 105,  66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
UPPER_LIP  = 13;  LOWER_LIP = 14

FILLER_WORDS = [
    "um","uh","umm","uhh","er","err","ah",
    "like","you know","i mean","basically",
    "i don't know","kind of","sort of",
]

# ─────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available(): return "mps"
    elif torch.cuda.is_available():       return "cuda"
    return "cpu"

# ─────────────────────────────────────────────
# CALGARY-CAMBRIDGE + TOPSIS
# ─────────────────────────────────────────────
CRITERIA = {
    "eye_contact"     : {"weight":0.25,"ideal":100.0,"worst":0.0,  "label":"Eye Contact",   "dimension":"Rapport Building",        "icon":"👁️"},
    "speech_clarity"  : {"weight":0.20,"ideal":0.0,  "worst":20.0, "label":"Speech Clarity","dimension":"Information Giving",       "icon":"🎙️"},
    "turn_balance"    : {"weight":0.25,"ideal":50.0, "worst":90.0, "label":"Turn Balance",  "dimension":"Gathering Information",    "icon":"👂"},
    "response_latency": {"weight":0.15,"ideal":0.5,  "worst":10.0, "label":"Patient Comfort","dimension":"Initiating Session",      "icon":"⏱️"},
    "body_language"   : {"weight":0.15,"ideal":10.0, "worst":0.0,  "label":"Body Language", "dimension":"Non-Verbal Communication","icon":"🙆"},
}

def calculate_topsis_score(visual_stats, speaker_stats):
    raw = {
        "eye_contact"     : visual_stats.get("eye_contact_pct", 0),
        "speech_clarity"  : speaker_stats.get("doctor", {}).get("filler_count", None),
        "turn_balance"    : speaker_stats.get("doctor_turn_pct", None),
        "response_latency": speaker_stats.get("avg_response_latency", None),
        "body_language"   : len(visual_stats.get("nod_timestamps", [])),
    }
    neutral = {"speech_clarity": 10, "turn_balance": 65, "response_latency": 2.0}
    for key in raw:
        if raw[key] is None:
            raw[key] = neutral.get(key, 0)

    normalized = {}
    for key, val in raw.items():
        ideal = CRITERIA[key]["ideal"]; worst = CRITERIA[key]["worst"]
        rang  = abs(ideal - worst)
        normalized[key] = 100.0 if rang == 0 else min(round(abs(val - worst) / rang * 100, 1), 100.0)

    weighted   = {k: normalized[k] * CRITERIA[k]["weight"] for k in normalized}
    d_plus     = np.sqrt(sum((CRITERIA[k]["weight"] * (normalized[k] - 100)) ** 2 for k in normalized))
    d_minus    = np.sqrt(sum((CRITERIA[k]["weight"] * normalized[k]) ** 2 for k in normalized))
    topsis_raw = d_minus / (d_plus + d_minus) if (d_plus + d_minus) > 0 else 0
    score      = round(topsis_raw * 100, 1)
    grade      = ("Excellent" if score >= 80 else "Good" if score >= 65
                  else "Fair" if score >= 45 else "Needs Improvement")

    breakdown = {}
    for key in normalized:
        val    = normalized[key]
        status = ("✅","score-excellent") if val>=80 else ("✅","score-good") if val>=60 else ("⚠️","score-fair") if val>=40 else ("❌","score-poor")
        breakdown[key] = {
            "raw_value": raw[key], "normalized": normalized[key],
            "weighted": round(weighted[key], 2), "weight_pct": f"{int(CRITERIA[key]['weight']*100)}%",
            "dimension": CRITERIA[key]["dimension"], "label": CRITERIA[key]["label"],
            "icon": CRITERIA[key]["icon"], "status_icon": status[0], "css_class": status[1],
        }
    return {"topsis_score":score,"grade":grade,"d_plus":round(d_plus,4),
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
        fillcolor="rgba(56,161,105,0.05)", line=dict(color="#38a169",width=1.5,dash="dot"), name="Ideal"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,100],tickfont=dict(size=10),gridcolor="#e2e8f0"),
                   angularaxis=dict(tickfont=dict(size=11,color="#2d3748")), bgcolor="white"),
        showlegend=True, legend=dict(orientation="h",y=-0.15),
        margin=dict(t=20,b=40,l=40,r=40), paper_bgcolor="white", height=340)
    return fig

# ─────────────────────────────────────────────
# V4 METRICS
# ─────────────────────────────────────────────
def calculate_patient_engagement(utterances, speaker_map):
    """Patient-initiated elaboration rate — brief Task 4."""
    if not utterances:
        return {"score":0,"words_per_turn":0,"questions_asked":0,
                "elaboration_ratio":0,"long_turns":0,"grade":"No data"}
    pl = speaker_map.get("patient","SPEAKER_01")
    dl = speaker_map.get("doctor","SPEAKER_00")
    pu = [u for u in utterances if u["speaker"]==pl]
    du = [u for u in utterances if u["speaker"]==dl]
    if not pu:
        return {"score":0,"words_per_turn":0,"questions_asked":0,
                "elaboration_ratio":0,"long_turns":0,"grade":"No data"}
    pat_w = sum(u.get("words",len(u["text"].split())) for u in pu)
    doc_w = sum(u.get("words",len(u["text"].split())) for u in du)
    wpt   = round(pat_w / max(len(pu),1), 1)
    qs    = sum(u["text"].count("?") for u in pu)
    ratio = round(pat_w / max(doc_w,1), 2)
    longs = sum(1 for u in pu if u.get("words",len(u["text"].split())) > 20)
    score = round(min(wpt/30*100,100)*0.4 + min(ratio/0.6*100,100)*0.4 + min(longs/max(len(pu),1)*200,100)*0.2, 1)
    grade = "High" if score>=70 else "Moderate" if score>=40 else "Low"
    return {"score":score,"words_per_turn":wpt,"questions_asked":qs,
            "elaboration_ratio":ratio,"long_turns":longs,"total_turns":len(pu),"grade":grade}

def calculate_hesitation_windows(speaker_stats, threshold=2.0):
    """Patient pauses flagged with clinical context — brief Task 2."""
    windows = []
    for lat in speaker_stats.get("response_latencies",[]):
        if lat["latency_s"] >= threshold:
            sev = "high" if lat["latency_s"]>=5 else "medium" if lat["latency_s"]>=3 else "low"
            windows.append({
                "time"    : lat["after_doctor_end"],
                "latency_s": lat["latency_s"],
                "context" : lat.get("doctor_said","")[:60],
                "severity": sev,
                "color"   : "#e53e3e" if sev=="high" else "#d69e2e" if sev=="medium" else "#3182ce",
            })
    return sorted(windows, key=lambda x: -x["latency_s"])

def calculate_brow_furrow_index(visual_stats):
    """Brow furrow = patient confusion — brief Task 1."""
    expr  = visual_stats.get("expr_counts",{})
    total = sum(expr.values())
    furr  = expr.get("Concerned / Furrowed",0)
    if total == 0:
        return {"pct":0,"count":0,"score":0,"interpretation":"No data"}
    pct   = round(furr/total*100, 1)
    score = round(max(0, 100-pct*3), 1)
    interp = ("Minimal confusion detected" if pct<10
              else "Some confusion — consider simpler language" if pct<25
              else "High confusion — simplify significantly")
    return {"pct":pct,"count":furr,"score":score,"interpretation":interp}

def calculate_session_arc(utterances, speaker_map):
    """First half vs second half quality — brief Task 5."""
    if len(utterances) < 4:
        return {"available":False}
    mid = len(utterances)//2
    dl  = speaker_map.get("doctor","SPEAKER_00")
    pl  = speaker_map.get("patient","SPEAKER_01")

    def half_stats(utts):
        pu = [u for u in utts if u["speaker"]==pl]
        du = [u for u in utts if u["speaker"]==dl]
        pw = sum(u.get("words",0) for u in pu)
        dw = sum(u.get("words",0) for u in du)
        fc = sum(sum(1 for fw in FILLER_WORDS if fw in u["text"].lower()) for u in du)
        return {"pat_ratio":round(pw/max(pw+dw,1)*100,1),"fillers":fc}

    h1 = half_stats(utterances[:mid])
    h2 = half_stats(utterances[mid:])
    delta   = round(h2["pat_ratio"]-h1["pat_ratio"], 1)
    f_delta = h2["fillers"]-h1["fillers"]
    arc     = round(max(0,min(100, 50 + delta*2 - max(f_delta*3,0))), 1)
    trend   = "↗ Improving" if delta>2 or f_delta<0 else "↘ Declining" if arc<40 else "→ Steady"
    return {"available":True,"first_half":h1,"second_half":h2,
            "pat_ratio_delta":delta,"filler_delta":f_delta,
            "arc_score":arc,"trend":trend,"improving":delta>2 or f_delta<0}

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
# AUDIO EXTRACTION — MP3 for Whisper, WAV for PyAnnote
# ─────────────────────────────────────────────
def extract_audio(video_path, output_dir):
    """
    Two files, two tools:
    MP3 → Groq Whisper  (transcription — compression fine)
    WAV → PyAnnote      (diarization  — MUST be uncompressed PCM
                         MP3 destroys voice fingerprints)
    """
    mp3_path = os.path.join(output_dir,"audio.mp3")
    wav_path = os.path.join(output_dir,"audio.wav")
    r1=subprocess.run(["ffmpeg","-i",video_path,"-vn","-ar","16000","-ac","1","-b:a","64k","-y",mp3_path],
                      capture_output=True,text=True,timeout=120)
    if r1.returncode!=0: raise RuntimeError(f"ffmpeg mp3 failed: {r1.stderr[:200]}")
    r2=subprocess.run(["ffmpeg","-i",video_path,"-vn","-ar","16000","-ac","1","-acodec","pcm_s16le","-y",wav_path],
                      capture_output=True,text=True,timeout=120)
    if r2.returncode!=0: raise RuntimeError(f"ffmpeg wav failed: {r2.stderr[:200]}")
    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path)<1000:
        raise RuntimeError("Audio extraction produced empty file")
    return mp3_path, wav_path

# ─────────────────────────────────────────────
# DIARIZATION — Whisper + PyAnnote
# ─────────────────────────────────────────────
def _merge_whisper_pyannote(whisper_segments, pyannote_segments):
    """3-strategy merge: overlap → midpoint → nearest."""
    if not pyannote_segments:
        return [{"speaker":"SPEAKER_00","start":int(s.get("start",0)*1000),
                 "end":int(s.get("end",0)*1000),"text":s.get("text","").strip(),
                 "confidence":s.get("avg_logprob",0)} for s in whisper_segments]
    utterances=[]
    for seg in whisper_segments:
        ss=seg.get("start",0); se=seg.get("end",0); sm=(ss+se)/2
        best="SPEAKER_00"; best_ov=-1
        for ps in pyannote_segments:
            ov=min(se,ps["end"])-max(ss,ps["start"])
            if ov>best_ov: best_ov=ov; best=ps["speaker"]
        if best_ov<=0:
            for ps in pyannote_segments:
                if ps["start"]<=sm<=ps["end"]: best=ps["speaker"]; break
            else:
                best=min(pyannote_segments,key=lambda ps:abs(((ps["start"]+ps["end"])/2)-sm))["speaker"]
        utterances.append({"speaker":best,"start":int(ss*1000),"end":int(se*1000),
                           "text":seg.get("text","").strip(),"confidence":seg.get("avg_logprob",0)})
    return utterances

def _heuristic_diarization(segments, full_text):
    utterances=[]; cs="A"
    for i,seg in enumerate(segments):
        txt=seg.get("text","").strip(); sms=int(seg.get("start",0)*1000); ems=int(seg.get("end",0)*1000)
        if i>0:
            gap=seg.get("start",0)-segments[i-1].get("end",0)
            if segments[i-1].get("text","").strip().endswith("?") or gap>0.8:
                cs="B" if cs=="A" else "A"
        utterances.append({"speaker":cs,"start":sms,"end":ems,"text":txt,"confidence":seg.get("avg_logprob",0)})
    return {"utterances":utterances,"text":full_text,"diarization_method":"heuristic"}

def transcribe_with_diarization(mp3_path, wav_path):
    groq_key=st.session_state.get("groq_key_store","")
    hf_token=st.session_state.get("hf_token_store","")
    device=get_device()
    headers={"Authorization":f"Bearer {groq_key}"}
    with open(mp3_path,"rb") as f:
        resp=requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                           headers=headers,
                           files={"file":(os.path.basename(mp3_path),f,"audio/mpeg")},
                           data={"model":"whisper-large-v3","response_format":"verbose_json",
                                 "language":"en","temperature":"0"},timeout=120)
    if resp.status_code==401: raise ValueError("Invalid Groq API key")
    if resp.status_code!=200: raise ValueError(f"Whisper failed {resp.status_code}: {resp.text[:200]}")
    wr=resp.json(); ws=wr.get("segments",[]); ft=wr.get("text","")
    if not hf_token:
        st.info("💡 Add HuggingFace token for PyAnnote diarization (~94% accuracy)")
        return _heuristic_diarization(ws,ft)
    try:
        from pyannote.audio import Pipeline
        pipe=Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token=hf_token)
        if device=="mps":   pipe=pipe.to(torch.device("mps"))
        elif device=="cuda":pipe=pipe.to(torch.device("cuda"))
        diar=pipe(wav_path,min_speakers=2,max_speakers=2)
        ps=[{"start":t.start,"end":t.end,"speaker":sp} for t,_,sp in diar.itertracks(yield_label=True)]
        st.caption(f"🔍 PyAnnote: {len(ps)} segments · {device.upper()}")
        return {"utterances":_merge_whisper_pyannote(ws,ps),"text":ft,
                "diarization_method":"pyannote","device_used":device.upper(),"segment_count":len(ps)}
    except ImportError:
        st.warning("⚠️ PyAnnote not installed — pip install pyannote.audio torch")
        return _heuristic_diarization(ws,ft)
    except Exception as e:
        st.warning(f"⚠️ PyAnnote failed: {str(e)[:150]} — heuristic fallback")
        return _heuristic_diarization(ws,ft)

# ─────────────────────────────────────────────
# SPEAKER IDENTIFICATION — 4 signals
# ─────────────────────────────────────────────
def identify_speakers(utterances):
    if not utterances: return {"doctor":"SPEAKER_00","patient":"SPEAKER_01"}
    all_sp=sorted({u["speaker"] for u in utterances})
    if len(all_sp)<2: return {"doctor":all_sp[0],"patient":"SPEAKER_01"}
    a,b=all_sp[0],all_sp[1]
    MED=["pain","symptom","medication","diagnosis","treatment","prescription",
         "blood","pressure","heart","breath","chest","fever","nausea","allergy",
         "history","condition","chronic","acute","dose","tablet","mg","ml",
         "examine","refer","specialist","follow up","test","result","scan",
         "recommend","advice","exercise","diet","anxiety","depression","sleep",
         "how long","when did","where does","describe","tell me","any other",
         "family history","surgical","operation","hospital","gp","doctor"]
    med={a:0,b:0}; qc={a:0,b:0}; wc={a:0,b:0}
    for u in utterances:
        sp=u["speaker"]; txt=u["text"].lower()
        if sp not in med: continue
        med[sp]+=sum(1 for t in MED if t in txt); qc[sp]+=txt.count("?")
        wc[sp]+=u.get("words",len(u["text"].split()))
    votes={a:0,b:0}
    for w in [max(med,key=med.get),max(qc,key=qc.get),utterances[0]["speaker"],max(wc,key=wc.get)]:
        if w in votes: votes[w]+=1
    dl=max(votes,key=votes.get); pl=b if dl==a else a
    return {"doctor":dl,"patient":pl}

def process_diarized_transcript(raw):
    return [{"speaker":u.get("speaker","?"),"start_s":round(u.get("start",0)/1000,1),
             "end_s":round(u.get("end",0)/1000,1),"text":u.get("text","").strip(),
             "confidence":round(u.get("confidence",0),2),"words":len(u.get("text","").split())}
            for u in raw.get("utterances",[])]

def analyze_per_speaker(utterances, speaker_map):
    dl=speaker_map["doctor"]; pl=speaker_map["patient"]
    du=[u for u in utterances if u["speaker"]==dl]; pu=[u for u in utterances if u["speaker"]==pl]
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
            if 0<=gap<15: latencies.append({"after_doctor_end":utt["end_s"],"patient_starts":nxt["start_s"],
                                             "latency_s":gap,"doctor_said":utt["text"][:60]})
    avg_lat=round(np.mean([l["latency_s"] for l in latencies]),2) if latencies else 0
    tt=len(utterances); dt=len(du); pt=len(pu)
    return {"doctor":{**stats(du),"label":dl},"patient":{**stats(pu),"label":pl},
            "response_latencies":latencies[:15],"avg_response_latency":avg_lat,
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
                        "speech":utt["text"][:120],"eye_contact":eye,"expression":dom_expr,"head_pose":dom_yaw})
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
# GROQ LLM — includes empathy scoring
# ─────────────────────────────────────────────
GROQ_CHAT_URL="https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   ="llama-3.3-70b-versatile"

def build_prompt(visual_stats, speaker_stats, utterances, speaker_map,
                 aligned, topsis, engagement, hesitation_windows):
    gaze_str=", ".join(f"{s}s-{e}s" for s,e in visual_stats["gaze_away_windows"][:5]) or "None"
    nod_str=", ".join(f"{t}s" for t in visual_stats.get("nod_timestamps",[])[:6]) or "None"
    bd_str="\n".join(f"  {v['icon']} {v['label']} ({v['weight_pct']}): {v['normalized']}/100"
                     for k,v in topsis["breakdown"].items())
    doc=speaker_stats["doctor"]; pat=speaker_stats["patient"]
    hes_str="\n".join(f"  {w['time']}s: patient paused {w['latency_s']}s after \"{w['context']}...\""
                      for w in hesitation_windows[:4]) or "None detected"
    aligned_str="\n".join(f"  [{ev['time']}] {ev['role']}: \"{ev['speech'][:60]}\"\n    Eye: {ev['eye_contact']} | Expr: {ev['expression']}"
                          for ev in aligned[:8])
    # ── TOPSIS score is LOCKED — LLM must use this exact value ──────────
    locked_score = topsis["topsis_score"]
    return f"""You are an expert clinical communication coach analysing a doctor-patient interaction.
The overall score is {locked_score}/100 — you MUST use this exact value, do not change it.

━━━ TOPSIS SCORE (LOCKED) ━━━
Overall: {locked_score}/100 — {topsis['grade']}
{bd_str}

━━━ VISUAL ━━━
Eye Contact: {visual_stats['eye_contact_pct']}% | Nodding: {nod_str} | Gaze-Away: {gaze_str}

━━━ DOCTOR ━━━
Words: {doc['word_count']} | WPM: {doc['wpm']} | Fillers: {doc['filler_count']} | Turns: {doc['turn_count']}

━━━ PATIENT ━━━
Words: {pat['word_count']} | WPM: {pat['wpm']} | Turns: {pat['turn_count']}
Engagement grade: {engagement.get('grade','N/A')} | Words/turn: {engagement.get('words_per_turn',0)}

━━━ HESITATION WINDOWS ━━━
{hes_str}

━━━ TURN-TAKING ━━━
Doctor: {speaker_stats['doctor_turn_pct']}% | Patient: {speaker_stats['patient_turn_pct']}%
Avg patient latency: {speaker_stats['avg_response_latency']}s

━━━ ALIGNMENT ━━━
{aligned_str}

━━━ INSTRUCTIONS ━━━
CRITICAL: overall_score MUST be exactly {locked_score}. Coach the DOCTOR only.
empathy_score: rate 0-100 how empathetic, warm, patient-centred the doctor was.
Return ONLY valid JSON, no markdown fences:

{{
  "overall_score": {locked_score},
  "topsis_explanation": "<2 sentences>",
  "summary": "<3 sentences coaching the doctor>",
  "empathy_score": <0-100 integer>,
  "empathy_assessment": "<1-2 sentences on warmth and tone>",
  "hesitation_analysis": "<1-2 sentences interpreting patient hesitation>",
  "eye_contact_feedback": {{"score":{round(topsis['breakdown']['eye_contact']['normalized'])},"assessment":"<impact>","recommendations":["<tip>","<tip>","<tip>"]}},
  "doctor_speech_feedback": {{"score":{round(topsis['breakdown']['speech_clarity']['normalized'])},"assessment":"<impact>","recommendations":["<tip>","<tip>","<tip>"]}},
  "listening_feedback": {{"score":{round(topsis['breakdown']['turn_balance']['normalized'])},"assessment":"<impact>","recommendations":["<tip>","<tip>"]}},
  "body_language_feedback": {{"score":{round(topsis['breakdown']['body_language']['normalized'])},"assessment":"<impact>","recommendations":["<tip>","<tip>"]}},
  "patient_impact_analysis": {{"score":{round(topsis['breakdown']['response_latency']['normalized'])},"assessment":"<impact>","recommendations":["<tip>","<tip>"]}},
  "behavioural_insights": ["<insight>","<insight>","<insight>"],
  "priority_actions": [{{"rank":1,"action":"<change>","rationale":"<evidence>"}},{{"rank":2,"action":"<change>","rationale":"<why>"}},{{"rank":3,"action":"<change>","rationale":"<why>"}}],
  "strengths": ["<strength>","<strength>"],
  "coaching_tips": ["<tip>","<tip>","<tip>"]
}}""".strip()

def get_llm_feedback(visual_stats, speaker_stats, utterances, speaker_map,
                     aligned, topsis, engagement, hesitation_windows, groq_key):
    prompt=build_prompt(visual_stats,speaker_stats,utterances,speaker_map,
                        aligned,topsis,engagement,hesitation_windows)
    headers={"Authorization":f"Bearer {groq_key.strip()}","Content-Type":"application/json"}
    payload={"model":GROQ_MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":3000,"temperature":0}
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
                raw=raw[:last_valid+1] if last_valid>0 else raw
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
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value {color_class}">{value}</div>'
                f'<div class="sub">{sub}</div></div>',unsafe_allow_html=True)

def render_progress(label, pct, color="#3182ce"):
    st.markdown(f'<div style="margin-bottom:.5rem"><div style="display:flex;justify-content:space-between;'
                f'font-size:.85rem;color:#4a5568;margin-bottom:.25rem"><span>{label}</span>'
                f'<span><b>{pct}%</b></span></div><div class="progress-bar-wrap">'
                f'<div class="progress-bar-fill" style="width:{pct}%;background:{color}"></div></div></div>',
                unsafe_allow_html=True)

def render_feedback_block(title, icon, data):
    score=data.get("score",0)
    with st.expander(f"{icon} {title} — Score: {score}/100",expanded=False):
        st.markdown(f'<div style="background:#f7fafc;border-radius:8px;padding:1rem;margin-bottom:1rem;'
                    f'font-size:.9rem;color:#4a5568;line-height:1.6">{data.get("assessment","No data.")}</div>',
                    unsafe_allow_html=True)
        for r in data.get("recommendations",[]): st.markdown(f"- {r}")

def pinned_block(feedback, fb_key, topsis_key, topsis):
    # ALWAYS pins score to TOPSIS — never uses LLM score value
    block=dict(feedback.get(fb_key,{}))
    block["score"]=round(topsis["breakdown"][topsis_key]["normalized"])
    return block

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ API Keys")
    groq_key=st.text_input("Groq API Key",type="password",
        help="Free at console.groq.com",placeholder="gsk_...")
    if groq_key: st.session_state["groq_key_store"]=groq_key
    if groq_key and not groq_key.strip().startswith("gsk_"):
        st.warning("⚠️ Groq keys start with **gsk_**")

    st.divider()
    st.markdown("### 🤗 HuggingFace Token")
    hf_token=st.text_input("HuggingFace Token",type="password",
        help="Free at huggingface.co — for PyAnnote speaker diarization",
        placeholder="hf_...")
    if hf_token: st.session_state["hf_token_store"]=hf_token
    device=get_device()
    if hf_token:
        st.success(f"🎙️ PyAnnote active · ~94% accuracy")
        st.caption(f"Device: {device.upper()} · WAV · Mac M1 MPS")
    else:
        st.warning("⚠️ Heuristic diarization · ~60%")
        st.caption("Add HF token for accurate speaker separation")

    st.divider()
    st.markdown("### 🔄 Speaker Assignment")
    st.caption("If Doctor/Patient labels are swapped:")
    if st.button("🔄 Swap Doctor ↔ Patient",use_container_width=True):
        st.session_state["swap_speakers"]=not st.session_state.get("swap_speakers",False)
        # Invalidate cached results so re-analysis runs
        st.session_state.pop("analysis_results",None)
    if st.session_state.get("swap_speakers",False):
        st.success("✅ Swapped — click Analyze to apply")
    else:
        st.info("👤 Auto-detected")

    st.divider()
    st.markdown("### 🖥️ View Mode")
    # ── KEY FIX: view mode switch does NOT trigger re-analysis ──
    # Results are cached in session_state and read by both modes
    view_mode=st.radio("Select mode",["👨‍⚕️ Quick Feedback","🔬 Research Mode"],index=0)
    is_quick=view_mode=="👨‍⚕️ Quick Feedback"

    st.divider()
    st.markdown("""
    <div style="background:#f0fff4;border:1px solid #9ae6b4;border-radius:8px;
                padding:.8rem 1rem;font-size:.82rem;color:#276749">
        ⚙️ <b>v4 Pipeline:</b><br>
        • Groq Whisper → transcription (MP3)<br>
        • PyAnnote → diarization (WAV)<br>
        • TOPSIS + Calgary-Cambridge scoring<br>
        • Empathy score via Groq Llama<br>
        • Patient engagement analysis<br>
        • Hesitation window detection<br>
        • Brow furrow confusion index<br>
        • Session quality arc
    </div>""",unsafe_allow_html=True)
    st.divider()
    st.caption("v4 — Multi-Modal Clinical Communication Analyzer")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Doctor-Patient Communication Analyzer</h1>
    <p>v4 · MediaPipe + Groq Whisper + PyAnnote + Groq Llama 3.3 70B · TOPSIS Calgary-Cambridge</p>
</div>""",unsafe_allow_html=True)

col_url,col_btn=st.columns([5,1])
with col_url:
    youtube_url=st.text_input("YouTube URL",placeholder="https://www.youtube.com/watch?v=...",
                               label_visibility="collapsed")
with col_btn:
    analyze_btn=st.button("▶ Analyze",use_container_width=True)

st.markdown('<div class="info-box">💡 <b>Best results:</b> Clear audio, both speakers visible. '
            'Try: <b>"OSCE clinical communication"</b> or <b>"doctor patient consultation"</b></div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANALYSIS PIPELINE
# ─────────────────────────────────────────────
# KEY FIX: Results stored in st.session_state
# Switching view mode (Quick ↔ Research) does NOT re-run the analysis
# Both modes always read from the SAME cached session_state dict
# ─────────────────────────────────────────────

if analyze_btn and youtube_url:
    if not groq_key: st.error("⚠️ Enter your Groq API key in the sidebar."); st.stop()

    # Clear any previous results so fresh analysis starts clean
    st.session_state.pop("analysis_results", None)
    tmp_dir=tempfile.mkdtemp()

    # ── Download ──────────────────────────────
    with st.status("📥 Downloading video…",expanded=True) as status:
        try:
            video_path,title,vid_duration=download_youtube(youtube_url,tmp_dir)
            status.update(label=f"✅ Downloaded: **{title}** ({int(vid_duration)}s)",state="complete")
        except Exception as e: st.error(f"Download failed: {e}"); st.stop()

    # ── MediaPipe ─────────────────────────────
    prog_bar=st.progress(0,text="🔍 Analyzing frames with MediaPipe…")
    def update_prog(v): prog_bar.progress(v,text=f"🔍 Analyzing frames… {int(v*100)}%")
    with st.spinner(""):
        visual_stats=analyze_video(video_path,sample_rate=4,max_frames=500,progress_cb=update_prog)
    prog_bar.progress(1.0,text="✅ Visual analysis complete"); time.sleep(0.2); prog_bar.empty()

    # ── Whisper + PyAnnote ────────────────────
    utterances=[]; speaker_map={"doctor":"SPEAKER_00","patient":"SPEAKER_01"}
    speaker_stats={}; mp3_path=None; wav_path=None; diarization_method="unknown"

    with st.status("🎙️ Groq Whisper (MP3) + PyAnnote (WAV)…",expanded=True) as audio_status:
        try:
            mp3_path,wav_path=extract_audio(video_path,tmp_dir)
            raw=transcribe_with_diarization(mp3_path,wav_path)
            diarization_method=raw.get("diarization_method","unknown")
            utterances=process_diarized_transcript(raw)
            speaker_map=identify_speakers(utterances)
            if st.session_state.get("swap_speakers",False):
                speaker_map={"doctor":speaker_map["patient"],"patient":speaker_map["doctor"]}
            speaker_stats=analyze_per_speaker(utterances,speaker_map)
            doc=speaker_stats["doctor"]; pat=speaker_stats["patient"]
            badge=(f"PyAnnote ~94% · WAV · {raw.get('device_used','')} · {raw.get('segment_count',0)} segs"
                   if diarization_method=="pyannote" else "Heuristic ~60% — add HF token for accuracy")
            audio_status.update(
                label=(f"✅ Doctor: {doc['word_count']}w {doc['wpm']}WPM {doc['filler_count']}fillers | "
                       f"Patient: {pat['word_count']}w {pat['wpm']}WPM | "
                       f"Latency: {speaker_stats['avg_response_latency']}s | {badge}"),
                state="complete")
        except Exception as e:
            audio_status.update(label=f"⚠️ Audio failed: {e}",state="error")
            speaker_stats={"doctor":{"word_count":0,"wpm":0,"filler_count":None,"turn_count":0,"avg_turn_duration":0,"label":"SPEAKER_00"},
                           "patient":{"word_count":0,"wpm":0,"filler_count":0,"turn_count":0,"avg_turn_duration":0,"label":"SPEAKER_01"},
                           "response_latencies":[],"avg_response_latency":None,
                           "total_turns":0,"doctor_turn_pct":None,"patient_turn_pct":None}

    # ── V4 metrics ────────────────────────────
    with st.spinner("📊 Computing v4 metrics…"):
        engagement         = calculate_patient_engagement(utterances,speaker_map)
        hesitation_windows = calculate_hesitation_windows(speaker_stats)
        brow_furrow        = calculate_brow_furrow_index(visual_stats)
        session_arc        = calculate_session_arc(utterances,speaker_map)

    # ── TOPSIS scoring ────────────────────────
    with st.spinner("📐 Calculating TOPSIS + Calgary-Cambridge…"):
        topsis=calculate_topsis_score(visual_stats,speaker_stats)

    # ── Groq Llama feedback ───────────────────
    aligned=align_speech_with_face(utterances,visual_stats,speaker_map)
    feedback=None
    with st.spinner("🤖 Generating AI feedback + empathy score…"):
        try:
            feedback=get_llm_feedback(visual_stats,speaker_stats,utterances,speaker_map,
                                      aligned,topsis,engagement,hesitation_windows,groq_key)
        except Exception as e:
            st.error(f"❌ AI feedback error: {e}")

    # ── SCORE LOCK ────────────────────────────
    # overall_score is ALWAYS topsis["topsis_score"]
    # LLM result is overwritten here regardless of what LLM returned
    # Both Quick mode and Research mode read from topsis["topsis_score"] directly
    # feedback["overall_score"] is set here only for JSON export — never used for display
    if feedback:
        feedback["overall_score"]=topsis["topsis_score"]   # lock
        feedback["overall_grade"]=topsis["grade"]          # lock

    # ── STORE IN SESSION STATE ────────────────
    # This is the KEY FIX for the "different score in different modes" bug:
    # All results stored once here. Switching Quick ↔ Research just changes
    # which view renders — the underlying data never changes.
    st.session_state["analysis_results"]={
        "title"             : title,
        "topsis"            : topsis,
        "visual_stats"      : visual_stats,
        "speaker_stats"     : speaker_stats,
        "utterances"        : utterances,
        "speaker_map"       : speaker_map,
        "feedback"          : feedback,
        "engagement"        : engagement,
        "hesitation_windows": hesitation_windows,
        "brow_furrow"       : brow_furrow,
        "session_arc"       : session_arc,
        "diarization_method": diarization_method,
        "aligned"           : aligned,
    }
    # Cleanup temp files
    try:
        for f in [video_path,mp3_path,wav_path]:
            if f and os.path.exists(f): os.remove(f)
        os.rmdir(tmp_dir)
    except Exception: pass

# ─────────────────────────────────────────────
# DISPLAY — reads from session_state
# Both Quick and Research modes use the SAME data
# Switching modes never re-runs analysis
# ─────────────────────────────────────────────
if "analysis_results" in st.session_state:
    R  = st.session_state["analysis_results"]
    topsis             = R["topsis"]
    visual_stats       = R["visual_stats"]
    speaker_stats      = R["speaker_stats"]
    utterances         = R["utterances"]
    speaker_map        = R["speaker_map"]
    feedback           = R["feedback"]
    engagement         = R["engagement"]
    hesitation_windows = R["hesitation_windows"]
    brow_furrow        = R["brow_furrow"]
    session_arc        = R["session_arc"]
    diarization_method = R["diarization_method"]
    title              = R["title"]

    # ── The ONE true score source ─────────────
    # topsis["topsis_score"] is the single source of truth
    # displayed identically in BOTH Quick and Research modes
    OVERALL_SCORE = topsis["topsis_score"]
    OVERALL_GRADE = topsis["grade"]

    st.success(f"✅ Analysis ready: **{title}**")
    doc=speaker_stats.get("doctor",{}); pat=speaker_stats.get("patient",{})

    # ── SHARED SCORE BANNER — shown identically in BOTH modes ─────────────
    # This is ONE piece of code. OVERALL_SCORE cannot differ between modes
    # because it is set once from session_state above and never changes.
    score_color=("#38a169" if OVERALL_SCORE>=80 else "#3182ce" if OVERALL_SCORE>=65
                 else "#d69e2e" if OVERALL_SCORE>=45 else "#e53e3e")
    st.markdown(
        f'<div style="background:white;border-radius:16px;padding:1.5rem 2rem;'
        f'border:2px solid {score_color}22;box-shadow:0 4px 16px rgba(0,0,0,0.08);'
        f'text-align:center;margin-bottom:1rem">'
        f'<div style="font-size:.75rem;color:#718096;font-weight:600;text-transform:uppercase;letter-spacing:.5px">'
        f'TOPSIS + Calgary-Cambridge · Overall Score</div>'
        f'<div style="font-size:3.5rem;font-weight:800;color:{score_color};line-height:1.1;margin:.3rem 0">'
        f'{OVERALL_SCORE}/100</div>'
        f'<div style="font-size:1.1rem;font-weight:600;color:{score_color}">{OVERALL_GRADE}</div>'
        f'<div style="font-size:.78rem;color:#a0aec0;margin-top:.3rem">',
        unsafe_allow_html=True)

    # Debug line — remove after confirming scores match
    st.caption(f"🔒 Score source: session_state · TOPSIS d+={topsis['d_plus']} d−={topsis['d_minus']} · Mode: {'Quick' if is_quick else 'Research'}")

    st.divider()

    # ══════════════════════════════════════════
    # QUICK MODE
    # ══════════════════════════════════════════
    if is_quick:
        st.markdown('<span class="mode-badge mode-quick">👨‍⚕️ Quick Feedback</span>',unsafe_allow_html=True)

        # ── Quick mode: empathy + engagement below the shared banner ──
        emp=feedback.get("empathy_score",0) if feedback else 0
        emp_col=("#38a169" if emp>=75 else "#3182ce" if emp>=50 else "#d69e2e" if emp>=30 else "#e53e3e")
        ec1,ec2=st.columns(2)
        with ec1:
            st.markdown(f'<div class="metric-card"><div class="label">Empathy Score</div>'
                        f'<div class="value" style="color:{emp_col}">{emp}/100</div>'
                        f'<div class="sub">Doctor warmth + tone</div></div>',unsafe_allow_html=True)
        with ec2:
            eng_col=("#38a169" if engagement["score"]>=70 else "#d69e2e" if engagement["score"]>=40 else "#e53e3e")
            st.markdown(f'<div class="metric-card"><div class="label">Patient Engagement</div>'
                        f'<div class="value" style="color:{eng_col}">{engagement["grade"]}</div>'
                        f'<div class="sub">{engagement["words_per_turn"]} words/turn</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("#### Calgary-Cambridge Dimensions")
        for key,val in topsis["breakdown"].items():
            n=val["normalized"]; col="#38a169" if n>=70 else "#d69e2e" if n>=40 else "#e53e3e"
            st.markdown(
                f'<div style="background:white;border-radius:10px;padding:.8rem 1.2rem;'
                f'border:1px solid #e2e8f0;margin-bottom:.5rem;display:flex;'
                f'align-items:center;justify-content:space-between">'
                f'<div><span style="font-size:1.1rem">{val["icon"]}</span>'
                f'<span style="font-weight:600;color:#2d3748;margin-left:.5rem">{val["label"]}</span>'
                f'<span style="font-size:.78rem;color:#718096;margin-left:.5rem">{val["dimension"]}</span></div>'
                f'<div style="font-size:1.2rem;font-weight:800;color:{col}">{val["status_icon"]} {round(n)}/100</div></div>',
                unsafe_allow_html=True)

        st.markdown("#### Performance Radar")
        st.plotly_chart(radar_chart(topsis),use_container_width=True)

        if feedback and feedback.get("summary"):
            st.info(f"📋 **Summary:** {feedback['summary']}")
        if feedback and feedback.get("empathy_assessment"):
            st.markdown(f'<div class="engagement-card">💚 <b>Empathy:</b> {feedback["empathy_assessment"]}</div>',
                        unsafe_allow_html=True)

        if feedback and feedback.get("priority_actions"):
            st.markdown("#### 🎯 Top 3 Actions")
            cols=st.columns(3)
            for action,col in zip(feedback["priority_actions"][:3],cols):
                rank=action.get("rank",1); color=["#e53e3e","#d69e2e","#3182ce"][min(rank-1,2)]
                with col:
                    st.markdown(
                        f'<div style="background:white;border-left:4px solid {color};border-radius:8px;'
                        f'padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,.06);height:100%">'
                        f'<div style="font-size:.75rem;font-weight:700;color:{color};text-transform:uppercase">#{rank}</div>'
                        f'<div style="font-size:.9rem;font-weight:600;color:#2d3748;margin:.3rem 0">{action.get("action","")}</div>'
                        f'<div style="font-size:.8rem;color:#718096">{action.get("rationale","")}</div></div>',
                        unsafe_allow_html=True)

        if feedback and feedback.get("strengths"):
            st.markdown("#### ✅ Strengths")
            for s in feedback["strengths"]:
                st.markdown(f"<span class='tag tag-green'>✓ {s}</span>",unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # RESEARCH MODE
    # ══════════════════════════════════════════
    else:
        st.markdown('<span class="mode-badge mode-research">🔬 Research Mode</span>',unsafe_allow_html=True)

        # TOPSIS explanation from LLM (score shown in shared banner above)
        if feedback and feedback.get("topsis_explanation"):
            st.markdown(
                f'<div class="topsis-box">'
                f'📐 <b>TOPSIS Explanation:</b> {feedback["topsis_explanation"]}<br>'
                f'd+: <b>{topsis["d_plus"]}</b> (distance from ideal) &nbsp;|&nbsp; '
                f'd−: <b>{topsis["d_minus"]}</b> (distance from worst)</div>',
                unsafe_allow_html=True)

        emp=feedback.get("empathy_score",0) if feedback else 0
        cols=st.columns(8)
        metrics=[
            ("Eye Contact",  f"{visual_stats['eye_contact_pct']}%","strong gaze",                  score_class(int(visual_stats['eye_contact_pct']))),
            ("Doctor WPM",   str(doc.get("wpm",0)),     "speech rate",                             "score-good" if 110<=doc.get("wpm",0)<=160 else "score-fair"),
            ("Patient WPM",  str(pat.get("wpm",0)),     "speech rate",                             "score-good" if pat.get("wpm",0)>50 else "score-fair"),
            ("Dr Fillers",   str(doc.get("filler_count",0)),"filler words",                        "score-good" if doc.get("filler_count",0)<5 else "score-fair" if doc.get("filler_count",0)<15 else "score-poor"),
            ("Avg Latency",  f"{speaker_stats.get('avg_response_latency',0) or 0}s","patient",     "score-good" if (speaker_stats.get("avg_response_latency",0) or 0)<2 else "score-fair"),
            ("Turn Balance", f"{speaker_stats.get('doctor_turn_pct',0)}%","doctor turns",          "score-good" if 40<=speaker_stats.get("doctor_turn_pct",0)<=60 else "score-fair"),
            ("Empathy",      f"{emp}/100",              "warmth",                                  score_class(emp)),
        ]
        cols=st.columns(7)
        for col,(lbl,val,sub,cls) in zip(cols,metrics):
            with col: render_metric(lbl,val,sub,cls)

        st.markdown("<br>",unsafe_allow_html=True)
        rc,bc=st.columns(2)
        with rc:
            st.markdown("#### 📊 Calgary-Cambridge Radar")
            st.plotly_chart(radar_chart(topsis),use_container_width=True)
        with bc:
            st.markdown("#### 📐 TOPSIS Breakdown")
            for key,val in topsis["breakdown"].items():
                n=val["normalized"]; col="#38a169" if n>=70 else "#d69e2e" if n>=40 else "#e53e3e"
                st.markdown(
                    f'<div style="background:white;border-radius:8px;padding:.7rem 1rem;'
                    f'border:1px solid #e2e8f0;margin-bottom:.5rem">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem">'
                    f'<span style="font-weight:600;font-size:.88rem;color:#2d3748">'
                    f'{val["icon"]} {val["label"]} ({val["weight_pct"]})</span>'
                    f'<span style="font-weight:800;color:{col}">{round(n)}/100</span></div>'
                    f'<div style="background:#edf2f7;border-radius:999px;height:6px;overflow:hidden">'
                    f'<div style="width:{n}%;height:100%;background:{col};border-radius:999px"></div></div>'
                    f'<div style="font-size:.74rem;color:#718096;margin-top:.3rem">'
                    f'{val["dimension"]} · Raw: {val["raw_value"]} · Weighted: {val["weighted"]}</div></div>',
                    unsafe_allow_html=True)

        st.divider()

        if feedback and feedback.get("summary"):
            st.markdown(f'<div class="feedback-section"><h3>📋 Executive Summary</h3>'
                        f'<p style="color:#4a5568;line-height:1.7;font-size:.95rem">{feedback["summary"]}</p></div>',
                        unsafe_allow_html=True)

        # Empathy
        if feedback and feedback.get("empathy_score") is not None:
            es=feedback.get("empathy_score",0)
            ecol="#38a169" if es>=75 else "#3182ce" if es>=50 else "#d69e2e" if es>=30 else "#e53e3e"
            st.markdown("#### 💚 Empathy Score")
            ea,eb=st.columns([1,3])
            with ea:
                st.markdown(f'<div class="metric-card"><div class="label">Empathy</div>'
                            f'<div class="value" style="color:{ecol}">{es}/100</div>'
                            f'<div class="sub">warmth + tone</div></div>',unsafe_allow_html=True)
            with eb:
                st.markdown(f'<div class="engagement-card" style="margin-top:.5rem">'
                            f'<b>Assessment:</b> {feedback.get("empathy_assessment","")}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div style="background:#edf2f7;border-radius:999px;height:10px;overflow:hidden;margin-top:.5rem">'
                            f'<div style="width:{es}%;height:100%;background:{ecol};border-radius:999px"></div></div>',
                            unsafe_allow_html=True)

        st.divider()

        # Patient Engagement
        st.markdown("#### 🧑 Patient Engagement")
        pg1,pg2,pg3,pg4=st.columns(4)
        with pg1: render_metric("Engagement",engagement["grade"],f"score: {engagement['score']}/100",
            "score-good" if engagement["score"]>=70 else "score-fair" if engagement["score"]>=40 else "score-poor")
        with pg2: render_metric("Words/Turn",str(engagement["words_per_turn"]),"avg elaboration",
            "score-good" if engagement["words_per_turn"]>=20 else "score-fair")
        with pg3: render_metric("Pat. Questions",str(engagement["questions_asked"]),"curiosity signal",
            "score-good" if engagement["questions_asked"]>=2 else "score-fair")
        with pg4: render_metric("Elaboration",f"{engagement['elaboration_ratio']}x","pat/doc word ratio",
            "score-good" if engagement["elaboration_ratio"]>=0.4 else "score-fair")
        if feedback and feedback.get("hesitation_analysis"):
            st.markdown(f'<div class="arc-card" style="margin-top:.5rem">'
                        f'🔍 <b>Hesitation interpretation:</b> {feedback["hesitation_analysis"]}</div>',
                        unsafe_allow_html=True)

        st.divider()

        # Hesitation Windows
        st.markdown("#### ⏱️ Hesitation Windows")
        if hesitation_windows:
            st.caption(f"{len(hesitation_windows)} patient pause(s) ≥ 2s detected")
            for w in hesitation_windows[:6]:
                st.markdown(
                    f'<div class="hesitation-card">'
                    f'<span style="color:{w["color"]};font-weight:700">⚠ {w["latency_s"]}s pause</span>'
                    f' at {w["time"]}s — After: <i>"{w["context"]}..."</i>'
                    f'<span style="float:right;font-size:.78rem;color:{w["color"]}">{w["severity"].upper()}</span></div>',
                    unsafe_allow_html=True)
        else:
            st.success("✅ No significant hesitation — patient responded promptly")

        st.divider()

        # Brow Furrow
        st.markdown("#### 🤨 Brow Furrow — Confusion Signal")
        bf1,bf2=st.columns(2)
        with bf1:
            render_metric("Furrow %",f"{brow_furrow['pct']}%",f"{brow_furrow['count']} frames",
                "score-good" if brow_furrow["pct"]<10 else "score-fair" if brow_furrow["pct"]<25 else "score-poor")
        with bf2:
            bfc="#38a169" if brow_furrow["pct"]<10 else "#d69e2e" if brow_furrow["pct"]<25 else "#e53e3e"
            st.markdown(f'<div class="arc-card"><b>Interpretation:</b> {brow_furrow["interpretation"]}<br>'
                        f'<div style="background:#edf2f7;border-radius:999px;height:8px;overflow:hidden;margin-top:.5rem">'
                        f'<div style="width:{min(brow_furrow["pct"]*3,100)}%;height:100%;'
                        f'background:{bfc};border-radius:999px"></div></div></div>',unsafe_allow_html=True)

        st.divider()

        # Session Arc
        st.markdown("#### 📈 Session Quality Arc")
        if session_arc.get("available"):
            sa1,sa2,sa3=st.columns(3)
            with sa1: render_metric("Session Arc",session_arc["trend"],
                f"arc score: {session_arc['arc_score']}/100",
                "score-good" if session_arc["improving"] else "score-fair")
            with sa2:
                h1=session_arc["first_half"]
                st.markdown(f'<div class="metric-card"><div class="label">First Half</div>'
                            f'<div class="value score-fair">{h1["pat_ratio"]}%</div>'
                            f'<div class="sub">patient speech | {h1["fillers"]} fillers</div></div>',unsafe_allow_html=True)
            with sa3:
                h2=session_arc["second_half"]
                dc="#38a169" if session_arc["pat_ratio_delta"]>0 else "#e53e3e"
                st.markdown(f'<div class="metric-card"><div class="label">Second Half</div>'
                            f'<div class="value" style="color:{dc}">{h2["pat_ratio"]}%</div>'
                            f'<div class="sub">patient speech | {h2["fillers"]} fillers | '
                            f'Δ {session_arc["pat_ratio_delta"]:+.1f}%</div></div>',unsafe_allow_html=True)
            interp=("Patient opened up — good rapport building." if session_arc["improving"]
                    else "Patient became less talkative — use more open questions.")
            st.markdown(f'<div class="arc-card" style="margin-top:.8rem">📊 {interp}</div>',unsafe_allow_html=True)
        else:
            st.info("Need ≥ 4 turns for session arc analysis")

        st.divider()

        # Visual Analysis
        st.markdown("### 👁️ Visual Analysis")
        v1,v2=st.columns(2)
        with v1:
            total_expr=sum(visual_stats["expr_counts"].values())
            cmap={"Neutral":"#4299e1","Attentive / Raised brow":"#38a169",
                  "Speaking / Reacting":"#d69e2e","Concerned / Furrowed":"#e53e3e","No face detected":"#a0aec0"}
            for expr,cnt in sorted(visual_stats["expr_counts"].items(),key=lambda x:-x[1]):
                render_progress(expr,round(cnt/max(total_expr,1)*100),cmap.get(expr,"#718096"))
        with v2:
            if visual_stats["gaze_away_windows"]:
                st.markdown("#### ⚠️ Gaze-Away Moments")
                for s,e in visual_stats["gaze_away_windows"][:5]:
                    st.markdown(f'<div style="border-left:3px solid #4299e1;padding-left:1rem;margin-bottom:.8rem">'
                                f'<div style="font-size:.75rem;color:#4299e1;font-weight:600">⏱ {s}s – {e}s</div>'
                                f'<div style="font-size:.85rem;color:#4a5568">Reduced eye contact</div></div>',
                                unsafe_allow_html=True)
            if visual_stats.get("nod_timestamps"):
                nod_str=" · ".join(f"{t}s" for t in visual_stats["nod_timestamps"][:8])
                st.markdown(f'<div class="info-box">🔄 Nodding at: {nod_str}</div>',unsafe_allow_html=True)

        st.divider()

        # Detailed AI Feedback
        st.markdown("### 🔬 Detailed AI Feedback")
        if feedback:
            insights=feedback.get("behavioural_insights",[])
            if insights:
                st.markdown("#### 🔍 Behavioural Insights")
                for ins in insights:
                    st.markdown(f'<div style="background:#faf5ff;border-left:4px solid #9f7aea;'
                                f'border-radius:8px;padding:.8rem 1rem;margin-bottom:.5rem;'
                                f'font-size:.88rem;color:#553c9a">💡 {ins}</div>',unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)
            fc1,fc2=st.columns(2)
            with fc1:
                render_feedback_block("Eye Contact",     "👁️",pinned_block(feedback,"eye_contact_feedback",   "eye_contact",    topsis))
                render_feedback_block("Doctor's Speech", "🎙️",pinned_block(feedback,"doctor_speech_feedback", "speech_clarity",  topsis))
                render_feedback_block("Listening Skills","👂",pinned_block(feedback,"listening_feedback",      "turn_balance",    topsis))
            with fc2:
                render_feedback_block("Body Language",   "🙆",pinned_block(feedback,"body_language_feedback",  "body_language",   topsis))
                render_feedback_block("Patient Impact",  "💙",pinned_block(feedback,"patient_impact_analysis", "response_latency",topsis))

            st.markdown("### 🎯 Priority Actions")
            if feedback.get("priority_actions"):
                p1,p2,p3=st.columns(3)
                for action,col in zip(feedback["priority_actions"],[p1,p2,p3]):
                    rank=action.get("rank",1); color=["#e53e3e","#d69e2e","#3182ce"][min(rank-1,2)]
                    with col:
                        st.markdown(
                            f'<div style="background:white;border-left:4px solid {color};border-radius:8px;'
                            f'padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,.06)">'
                            f'<div style="font-size:.75rem;font-weight:700;color:{color};text-transform:uppercase">#{rank}</div>'
                            f'<div style="font-size:.9rem;font-weight:600;color:#2d3748;margin:.3rem 0">{action.get("action","")}</div>'
                            f'<div style="font-size:.8rem;color:#718096">{action.get("rationale","")}</div></div>',
                            unsafe_allow_html=True)

            if feedback.get("strengths"):
                st.markdown("#### ✅ Strengths")
                for s in feedback["strengths"]:
                    st.markdown(f"<span class='tag tag-green'>✓ {s}</span>",unsafe_allow_html=True)

            tips=feedback.get("coaching_tips",[])
            if tips:
                st.markdown("### 💡 Coaching Tips")
                tip_icons=["🗣️","👂","🤝","💬","🌡️"]; cols=st.columns(len(tips))
                for i,(tip,col) in enumerate(zip(tips,cols)):
                    with col:
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#ebf8ff,#e6fffa);'
                            f'border-radius:10px;padding:1rem;text-align:center;'
                            f'border:1px solid #bee3f8;height:100%">'
                            f'<div style="font-size:1.5rem">{tip_icons[i%len(tip_icons)]}</div>'
                            f'<div style="font-size:.85rem;color:#2c5282;margin-top:.5rem;line-height:1.5">{tip}</div></div>',
                            unsafe_allow_html=True)

        with st.expander("🗃️ Raw Analysis Data (JSON)"):
            st.json({
                "version":"v4","algorithm":"TOPSIS + Calgary-Cambridge",
                "score_source":"topsis[topsis_score] — immutable",
                "overall_score":OVERALL_SCORE,
                "diarization_method":diarization_method,
                "topsis_result":{k:v for k,v in topsis.items() if k!="breakdown"},
                "topsis_breakdown":{k:{"label":v["label"],"raw_value":v["raw_value"],
                                       "normalized":v["normalized"],"weighted":v["weighted"],
                                       "weight":v["weight_pct"],"dimension":v["dimension"]}
                                    for k,v in topsis["breakdown"].items()},
                "v4_metrics":{"patient_engagement":engagement,"hesitation_windows":hesitation_windows,
                              "brow_furrow_index":brow_furrow,
                              "session_arc":{k:v for k,v in session_arc.items() if k not in ["first_half","second_half"]},
                              "empathy_score":feedback.get("empathy_score") if feedback else None},
                "visual_stats":{k:v for k,v in visual_stats.items() if k!="frames_data"},
                "speaker_stats":{k:v for k,v in speaker_stats.items() if k!="response_latencies"},
                "ai_feedback":feedback,
            })

elif analyze_btn and not youtube_url:
    st.warning("Please enter a YouTube URL to analyze.")

st.markdown(
    '<div style="text-align:center;color:#a0aec0;font-size:.8rem;margin-top:3rem;'
    'padding:1rem;border-top:1px solid #e2e8f0">'
    'v4 · Groq Whisper (MP3) + PyAnnote (WAV · Mac M1 MPS) + Groq Llama 3.3 70B · '
    'TOPSIS + Calgary-Cambridge · Empathy · Engagement · Hesitation · Brow Furrow · Session Arc'
    '</div>',unsafe_allow_html=True)
