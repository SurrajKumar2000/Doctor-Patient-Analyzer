import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
import time
import subprocess
import requests
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
.metric-card .label { font-size:.8rem; color:#718096; font-weight:500; text-transform:uppercase; letter-spacing:.5px; }
.metric-card .value { font-size:2rem; font-weight:700; margin:.4rem 0 .2rem; }
.metric-card .sub   { font-size:.8rem; color:#a0aec0; }

.score-excellent { color:#38a169; }
.score-good      { color:#3182ce; }
.score-fair      { color:#d69e2e; }
.score-poor      { color:#e53e3e; }

.feedback-section {
    background: white; border-radius: 12px; padding: 1.5rem;
    border: 1px solid #e2e8f0; margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.tag { display:inline-block; padding:.25rem .7rem; border-radius:999px; font-size:.75rem; font-weight:600; margin:.2rem; }
.tag-green  { background:#c6f6d5; color:#276749; }
.tag-yellow { background:#fefcbf; color:#975a16; }
.tag-red    { background:#fed7d7; color:#9b2c2c; }

.progress-bar-wrap { background:#edf2f7; border-radius:999px; height:10px; margin:.4rem 0 1rem; overflow:hidden; }
.progress-bar-fill { height:100%; border-radius:999px; transition:width .6s ease; }

.timeline-item { border-left: 3px solid #4299e1; padding-left: 1rem; margin-bottom: 1rem; }
.timeline-time { font-size:.75rem; color:#4299e1; font-weight:600; }
.timeline-text { font-size:.85rem; color:#4a5568; margin-top:.2rem; }

.transcript-box {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:1rem; font-size:.85rem; color:#4a5568; line-height:1.8;
    max-height:200px; overflow-y:auto;
}
.filler-word { background:#fed7d7; color:#9b2c2c; border-radius:3px; padding:0 3px; font-weight:600; }
.silence-tag { background:#fefcbf; color:#975a16; border-radius:3px; padding:0 3px; font-size:.75rem; }

.stButton>button {
    background: linear-gradient(135deg,#2b6cb0,#2c7a7b); color: white;
    border: none; padding: .6rem 2rem; border-radius: 8px;
    font-weight: 600; font-size: .95rem; transition: all .2s; width: 100%;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 4px 12px rgba(43,108,176,.4); }

.info-box {
    background:#ebf8ff; border:1px solid #90cdf4; border-radius:8px;
    padding:1rem 1.2rem; color:#2c5282; font-size:.9rem;
}
.warn-box {
    background:#fffaf0; border:1px solid #f6ad55; border-radius:8px;
    padding:1rem 1.2rem; color:#7b341e; font-size:.9rem;
}
.audio-badge {
    background:#e9d8fd; border:1px solid #d6bcfa; border-radius:8px;
    padding:.5rem 1rem; color:#553c9a; font-size:.85rem; font-weight:600;
    display:inline-block; margin-bottom:.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP (0.10.13+ compatible)
# ─────────────────────────────────────────────
from mediapipe.python.solutions import face_mesh as _mp_face_mesh_mod

class _FaceMeshNS:
    FaceMesh = _mp_face_mesh_mod.FaceMesh

mp_face_mesh = _FaceMeshNS()

LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
NOSE_TIP  = 1
CHIN      = 152
LEFT_EAR  = 234
RIGHT_EAR = 454
LEFT_BROW  = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
UPPER_LIP = 13
LOWER_LIP = 14

FILLER_WORDS = [
    "um", "uh", "umm", "uhh", "er", "err", "ah", "ahh",
    "like", "you know", "i mean", "basically", "literally",
    "i don't know", "kind of", "sort of", "right", "okay so"
]

# ─────────────────────────────────────────────
# MEDIAPIPE ANALYSIS HELPERS
# ─────────────────────────────────────────────
def estimate_eye_contact(lm, w, h):
    nose  = lm.landmark[NOSE_TIP]
    left  = lm.landmark[LEFT_EAR]
    right = lm.landmark[RIGHT_EAR]
    chin  = lm.landmark[CHIN]
    top   = lm.landmark[10]
    face_width    = abs(left.x - right.x)
    face_center_x = (left.x + right.x) / 2
    face_center_y = (top.y + chin.y) / 2
    h_offset      = abs(face_center_x - 0.5) / max(face_width, 0.01)
    v_offset      = abs(face_center_y - 0.5)
    yaw_proxy     = abs(nose.x - face_center_x) / max(face_width, 0.01)
    score = max(0, 1 - h_offset * 2 - v_offset * 1.5 - yaw_proxy * 3)
    return float(np.clip(score, 0, 1))

def estimate_head_pose(lm):
    nose  = lm.landmark[NOSE_TIP]
    chin  = lm.landmark[CHIN]
    top   = lm.landmark[10]
    left  = lm.landmark[LEFT_EAR]
    right = lm.landmark[RIGHT_EAR]
    face_vert = chin.y - top.y
    nose_pos  = (nose.y - top.y) / max(face_vert, 0.01)
    pitch = "Head tilted up" if nose_pos < 0.40 else ("Head tilted down" if nose_pos > 0.60 else "Head level")
    center_x   = (left.x + right.x) / 2
    yaw_offset = nose.x - center_x
    yaw = "Turned left" if yaw_offset < -0.05 else ("Turned right" if yaw_offset > 0.05 else "Facing forward")
    return pitch, yaw

def estimate_engagement(lm):
    lb_y = np.mean([lm.landmark[i].y for i in LEFT_BROW])
    rb_y = np.mean([lm.landmark[i].y for i in RIGHT_BROW])
    le_y = np.mean([lm.landmark[i].y for i in LEFT_EYE_INDICES])
    re_y = np.mean([lm.landmark[i].y for i in RIGHT_EYE_INDICES])
    brow_eye_gap = ((le_y - lb_y) + (re_y - rb_y)) / 2
    mouth_open   = lm.landmark[LOWER_LIP].y - lm.landmark[UPPER_LIP].y
    brow_spread  = lm.landmark[336].x - lm.landmark[107].x
    return {"brow_raise": float(brow_eye_gap), "mouth_open": float(mouth_open), "brow_spread": float(brow_spread)}

def classify_expression(eng):
    if eng["mouth_open"] > 0.04:  return "Speaking / Reacting"
    if eng["brow_raise"] > 0.06:  return "Attentive / Raised brow"
    if eng["brow_spread"] < 0.07: return "Concerned / Furrowed"
    return "Neutral"

def detect_nod(prev_pitch_y, curr_pitch_y, threshold=0.015):
    """Simple nod detection via vertical nose movement between frames."""
    return abs(curr_pitch_y - prev_pitch_y) > threshold

# ─────────────────────────────────────────────
# VIDEO ANALYSIS (MediaPipe)
# ─────────────────────────────────────────────
def analyze_video(video_path, sample_rate=4, max_frames=500, progress_cb=None):
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    frames_data   = []
    face_detected = 0
    frame_idx     = 0
    sampled       = 0
    prev_nose_y   = None
    nod_events    = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh:
        while cap.isOpened() and sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                ts  = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                entry = {"timestamp": round(ts, 2), "face_found": False, "num_faces": 0}
                if res.multi_face_landmarks:
                    entry["face_found"] = True
                    entry["num_faces"]  = len(res.multi_face_landmarks)
                    face_detected += 1
                    lm = res.multi_face_landmarks[0]
                    entry["eye_contact_score"] = estimate_eye_contact(lm, w, h)
                    pitch, yaw                 = estimate_head_pose(lm)
                    entry["head_pitch"]        = pitch
                    entry["head_yaw"]          = yaw
                    eng                        = estimate_engagement(lm)
                    entry["expression"]        = classify_expression(eng)
                    # Nod detection
                    curr_nose_y = lm.landmark[NOSE_TIP].y
                    if prev_nose_y is not None and detect_nod(prev_nose_y, curr_nose_y):
                        nod_events.append(round(ts, 2))
                    prev_nose_y = curr_nose_y
                else:
                    entry["eye_contact_score"] = 0.0
                    entry["head_pitch"]        = "No face"
                    entry["head_yaw"]          = "No face"
                    entry["expression"]        = "No face detected"
                    prev_nose_y = None
                frames_data.append(entry)
                sampled += 1
                if progress_cb:
                    progress_cb(min(frame_idx / max(total, 1), 0.95))
            frame_idx += 1
    cap.release()

    detected       = [f for f in frames_data if f["face_found"]]
    detection_rate = len(detected) / max(len(frames_data), 1)
    eye_scores     = [f["eye_contact_score"] for f in detected]
    avg_eye        = float(np.mean(eye_scores)) if eye_scores else 0
    eye_pct        = float(np.mean([s > 0.45 for s in eye_scores])) if eye_scores else 0

    expr_counts = {}
    for f in detected:
        e = f["expression"]
        expr_counts[e] = expr_counts.get(e, 0) + 1

    head_yaws   = [f["head_yaw"] for f in detected]
    forward_pct = sum(1 for y in head_yaws if "forward" in y.lower()) / max(len(head_yaws), 1)

    # Gaze-away clustering
    gaze_events = [f["timestamp"] for f in detected if f["eye_contact_score"] < 0.3]
    gaze_windows = []
    if gaze_events:
        start = prev = gaze_events[0]
        for t in gaze_events[1:]:
            if t - prev > 3:
                gaze_windows.append((round(start), round(prev)))
                start = t
            prev = t
        gaze_windows.append((round(start), round(prev)))

    # Cluster nod events
    nod_windows = []
    if nod_events:
        start = prev = nod_events[0]
        for t in nod_events[1:]:
            if t - prev > 2:
                nod_windows.append(round(start))
                start = t
            prev = t
        nod_windows.append(round(start))

    return {
        "duration_sec"     : round(duration, 1),
        "total_frames"     : total,
        "sampled_frames"   : sampled,
        "detection_rate"   : round(detection_rate * 100, 1),
        "avg_eye_contact"  : round(avg_eye, 3),
        "eye_contact_pct"  : round(eye_pct * 100, 1),
        "forward_pct"      : round(forward_pct * 100, 1),
        "expr_counts"      : expr_counts,
        "gaze_away_windows": gaze_windows,
        "nod_timestamps"   : nod_windows[:20],
        "frames_data"      : frames_data,
        "fps"              : round(fps, 1),
        "resolution"       : f"{w}x{h}",
    }

# ─────────────────────────────────────────────
# AUDIO EXTRACTION
# ─────────────────────────────────────────────
def extract_audio(video_path, output_dir):
    """Extract audio from video as mp3 using ffmpeg."""
    audio_path = os.path.join(output_dir, "audio.mp3")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                    # no video
        "-ar", "16000",           # 16kHz sample rate (Whisper optimal)
        "-ac", "1",               # mono
        "-b:a", "64k",            # compress for faster upload
        "-y",                     # overwrite
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        raise RuntimeError("Audio extraction produced empty file")
    return audio_path

# ─────────────────────────────────────────────
# WHISPER TRANSCRIPTION via Groq
# ─────────────────────────────────────────────
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL    = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL       = "llama-3.3-70b-versatile"
WHISPER_MODEL    = "whisper-large-v3"

def transcribe_audio(audio_path, api_key):
    """Send audio to Groq Whisper and return transcript with segments."""
    headers = {"Authorization": f"Bearer {api_key.strip()}"}

    with open(audio_path, "rb") as f:
        files   = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
        data    = {
            "model"            : WHISPER_MODEL,
            "response_format"  : "verbose_json",   # gives us timestamps
            "language"         : "en",
        }
        resp = requests.post(
            GROQ_WHISPER_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if resp.status_code == 401:
        raise ValueError("Invalid Groq API key")
    if resp.status_code != 200:
        raise ValueError(f"Whisper API error {resp.status_code}: {resp.text[:300]}")

    return resp.json()

# ─────────────────────────────────────────────
# SPEECH ANALYSIS
# ─────────────────────────────────────────────
def analyze_speech(transcript_data):
    """
    Extract paralinguistic features from Whisper transcript:
    - Filler words with timestamps
    - Silence / pause windows
    - Speech rate (words per minute)
    - Hesitation markers
    """
    segments  = transcript_data.get("segments", [])
    full_text = transcript_data.get("text", "").strip()

    if not segments:
        return {
            "full_text"        : full_text,
            "word_count"       : len(full_text.split()),
            "speech_rate_wpm"  : 0,
            "filler_count"     : 0,
            "filler_instances" : [],
            "silence_windows"  : [],
            "hesitation_count" : 0,
            "segment_count"    : 0,
            "avg_segment_gap"  : 0,
        }

    # Speech rate
    total_words    = len(full_text.split())
    total_duration = segments[-1]["end"] if segments else 1
    speech_rate    = round((total_words / max(total_duration, 1)) * 60, 1)

    # Filler word detection
    filler_instances = []
    text_lower = full_text.lower()
    for fw in FILLER_WORDS:
        idx = 0
        while True:
            pos = text_lower.find(fw, idx)
            if pos == -1:
                break
            # Try to find timestamp from segments
            char_count = 0
            ts = None
            for seg in segments:
                seg_len = len(seg.get("text",""))
                if char_count + seg_len >= pos:
                    ts = round(seg.get("start", 0), 1)
                    break
                char_count += seg_len
            filler_instances.append({"word": fw, "timestamp": ts})
            idx = pos + len(fw)

    # Silence / pause detection (gaps between segments > 1.5s)
    silence_windows = []
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i-1]["end"]
        if gap > 1.5:
            silence_windows.append({
                "start": round(segments[i-1]["end"], 1),
                "end"  : round(segments[i]["start"], 1),
                "gap_s": round(gap, 1),
            })

    # Hesitation markers (incomplete sentences, trailing off)
    hesitation_count = sum(
        1 for seg in segments
        if seg.get("text","").strip().endswith(("...", "—", "-"))
        or "i don't know" in seg.get("text","").lower()
        or "i'm not sure" in seg.get("text","").lower()
    )

    # Average gap between segments
    if len(segments) > 1:
        gaps = [segments[i]["start"] - segments[i-1]["end"]
                for i in range(1, len(segments))]
        avg_gap = round(np.mean(gaps), 2)
    else:
        avg_gap = 0

    return {
        "full_text"        : full_text,
        "word_count"       : total_words,
        "speech_rate_wpm"  : speech_rate,
        "filler_count"     : len(filler_instances),
        "filler_instances" : filler_instances[:20],
        "silence_windows"  : silence_windows[:15],
        "hesitation_count" : hesitation_count,
        "segment_count"    : len(segments),
        "avg_segment_gap"  : avg_gap,
        "segments"         : segments[:30],   # first 30 for alignment
    }

# ─────────────────────────────────────────────
# FACIAL + SPEECH ALIGNMENT
# ─────────────────────────────────────────────
def align_speech_with_face(speech, visual):
    """
    Pair each speech segment with co-occurring facial signals.
    Returns list of aligned events for the prompt.
    """
    aligned = []
    segments = speech.get("segments", [])
    frames   = visual.get("frames_data", [])

    for seg in segments[:20]:   # limit to first 20 segments
        seg_start = seg.get("start", 0)
        seg_end   = seg.get("end", seg_start + 1)
        seg_text  = seg.get("text", "").strip()

        # Find face frames within this speech segment's time window
        matching = [
            f for f in frames
            if f.get("face_found") and seg_start <= f["timestamp"] <= seg_end
        ]

        if matching:
            eye_avg   = round(np.mean([f["eye_contact_score"] for f in matching]), 2)
            expressions = [f["expression"] for f in matching]
            dom_expr  = max(set(expressions), key=expressions.count)
            yaws      = [f["head_yaw"] for f in matching]
            dom_yaw   = max(set(yaws), key=yaws.count)
        else:
            eye_avg  = None
            dom_expr = "Unknown"
            dom_yaw  = "Unknown"

        # Check if silence occurred just before this segment
        silence_before = any(
            abs(sw["end"] - seg_start) < 0.5
            for sw in speech.get("silence_windows", [])
        )

        aligned.append({
            "time"          : f"{round(seg_start,1)}s-{round(seg_end,1)}s",
            "speech"        : seg_text[:120],
            "eye_contact"   : eye_avg,
            "expression"    : dom_expr,
            "head_pose"     : dom_yaw,
            "silence_before": silence_before,
        })

    return aligned

# ─────────────────────────────────────────────
# YOUTUBE DOWNLOAD
# ─────────────────────────────────────────────
def download_youtube(url, output_dir):
    import yt_dlp
    out_tmpl = os.path.join(output_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format"             : "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]",
        "outtmpl"            : out_tmpl,
        "quiet"              : True,
        "no_warnings"        : True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        for fn in Path(output_dir).iterdir():
            if fn.suffix in [".mp4", ".mkv", ".webm"]:
                return str(fn), info.get("title", "Unknown"), round(info.get("duration", 0), 0)
    raise FileNotFoundError("Download failed — no video file found.")

# ─────────────────────────────────────────────
# GROQ LLM FEEDBACK (multimodal prompt)
# ─────────────────────────────────────────────
def build_multimodal_prompt(visual_stats, speech_stats, aligned_events, url):
    gaze_str = ", ".join(
        f"{s}s-{e}s" for s, e in visual_stats["gaze_away_windows"][:8]
    ) or "None"

    expr_str = "\n".join(
        f"  - {expr}: {cnt} frames ({round(cnt/max(sum(visual_stats['expr_counts'].values()),1)*100)}%)"
        for expr, cnt in sorted(visual_stats["expr_counts"].items(), key=lambda x: -x[1])
    )

    nod_str = ", ".join(f"{t}s" for t in visual_stats.get("nod_timestamps", [])[:10]) or "None detected"

    silence_str = ", ".join(
        f"{sw['start']}s-{sw['end']}s ({sw['gap_s']}s gap)"
        for sw in speech_stats.get("silence_windows", [])[:6]
    ) or "None"

    filler_str = ", ".join(
        f"\"{fi['word']}\" at {fi['timestamp']}s"
        for fi in speech_stats.get("filler_instances", [])[:10]
        if fi["timestamp"] is not None
    ) or f"{speech_stats.get('filler_count', 0)} instances detected"

    aligned_str = "\n".join(
        f"  [{ev['time']}] \"{ev['speech']}\"\n"
        f"    → Eye contact: {ev['eye_contact']}, Expression: {ev['expression']}, "
        f"Head: {ev['head_pose']}, Silence before: {ev['silence_before']}"
        for ev in aligned_events[:12]
    ) or "No aligned data"

    return f"""You are an expert clinical communication coach analysing a doctor-patient interaction video.

You have BOTH visual (MediaPipe) AND speech (Whisper) data. Use both to give deeply contextual feedback.

━━━ VISUAL ANALYSIS ━━━
- Duration: {visual_stats['duration_sec']}s
- Face Detection Rate: {visual_stats['detection_rate']}%
- Eye Contact Score (avg): {visual_stats['avg_eye_contact']:.2f}/1.0
- Strong Eye Contact Frames: {visual_stats['eye_contact_pct']}%
- Head Facing Forward: {visual_stats['forward_pct']}%
- Gaze-Away Windows: {gaze_str}
- Nodding Detected At: {nod_str}

EXPRESSION DISTRIBUTION:
{expr_str}

━━━ SPEECH ANALYSIS (Whisper) ━━━
- Total Words Spoken: {speech_stats['word_count']}
- Speech Rate: {speech_stats['speech_rate_wpm']} words/min
- Filler Words: {speech_stats['filler_count']} instances ({filler_str})
- Silence/Pause Windows: {silence_str}
- Hesitation Markers: {speech_stats['hesitation_count']}
- Avg Gap Between Segments: {speech_stats['avg_segment_gap']}s

━━━ ALIGNED MULTIMODAL TIMELINE ━━━
(Each row = what was said + what face was doing simultaneously)
{aligned_str}

━━━ TRANSCRIPT EXCERPT ━━━
{speech_stats['full_text'][:600]}...

━━━ INSTRUCTIONS ━━━
Using BOTH the visual and speech data together, provide clinical communication feedback.
Cross-reference: e.g. did eye contact drop when filler words appeared? Did silence align with confused expression?
Return ONLY valid JSON, no markdown fences:

{{
  "overall_score": <0-100>,
  "overall_grade": "<Excellent|Good|Fair|Needs Improvement>",
  "summary": "<3 sentence summary referencing both visual and speech findings>",
  "eye_contact_feedback": {{
    "score": <0-100>,
    "assessment": "<paragraph referencing specific timestamps>",
    "recommendations": ["<tip>", "<tip>", "<tip>"]
  }},
  "speech_feedback": {{
    "score": <0-100>,
    "assessment": "<paragraph on speech rate, fillers, hesitation, silence patterns>",
    "filler_word_note": "<specific note on filler word usage>",
    "recommendations": ["<tip>", "<tip>", "<tip>"]
  }},
  "body_language_feedback": {{
    "score": <0-100>,
    "assessment": "<paragraph on head pose, nodding, engagement>",
    "recommendations": ["<tip>", "<tip>", "<tip>"]
  }},
  "expression_feedback": {{
    "score": <0-100>,
    "assessment": "<paragraph interpreting expressions in clinical context>",
    "recommendations": ["<tip>", "<tip>"]
  }},
  "patient_comfort_analysis": {{
    "score": <0-100>,
    "assessment": "<how visual + speech signals together affect patient comfort>",
    "key_moments": ["<moment: what speech + face showed simultaneously>"],
    "recommendations": ["<tip>", "<tip>"]
  }},
  "multimodal_insights": [
    "<insight where speech and face data correlate — e.g. filler words at same time as gaze aversion>",
    "<another cross-modal insight>",
    "<third insight>"
  ],
  "priority_actions": [
    {{"rank": 1, "action": "<action>", "rationale": "<why, referencing data>"}},
    {{"rank": 2, "action": "<action>", "rationale": "<why>"}},
    {{"rank": 3, "action": "<action>", "rationale": "<why>"}}
  ],
  "strengths": ["<strength 1>", "<strength 2>"],
  "clinical_communication_tips": ["<tip>", "<tip>", "<tip>"]
}}""".strip()


def get_llm_feedback(visual_stats, speech_stats, aligned_events, url, api_key):
    """Call Groq LLM with multimodal prompt."""
    prompt  = build_multimodal_prompt(visual_stats, speech_stats, aligned_events, url)
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type" : "application/json",
    }
    payload = {
        "model"      : GROQ_MODEL,
        "messages"   : [{"role": "user", "content": prompt}],
        "max_tokens" : 2000,
        "temperature": 0.3,
    }

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 401:
                raise ValueError("Invalid Groq API key")
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"): part = part[4:].strip()
                    if part.startswith("{"): raw = part; break
            return json.loads(raw)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            time.sleep(3 * (attempt + 1))
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse Groq response as JSON: {e}")
        except ValueError:
            raise
    raise ConnectionError(f"Could not reach Groq after 3 attempts. Last error: {last_error}")

# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────
def score_class(s):
    if s >= 80: return "score-excellent"
    if s >= 60: return "score-good"
    if s >= 40: return "score-fair"
    return "score-poor"

def render_metric(label, value, sub="", color_class="score-good"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {color_class}">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def render_progress(label, pct, color="#3182ce"):
    st.markdown(f"""
    <div style="margin-bottom:.5rem">
        <div style="display:flex;justify-content:space-between;font-size:.85rem;color:#4a5568;margin-bottom:.25rem">
            <span>{label}</span><span><b>{pct}%</b></span>
        </div>
        <div class="progress-bar-wrap">
            <div class="progress-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
    </div>""", unsafe_allow_html=True)

def render_feedback_block(title, icon, data, extra_key=None):
    score = data.get("score", 0)
    with st.expander(f"{icon} {title} — Score: {score}/100", expanded=True):
        st.markdown(f"""
        <div style="background:#f7fafc;border-radius:8px;padding:1rem;margin-bottom:1rem;
                    font-size:.9rem;color:#4a5568;line-height:1.6">
        {data.get('assessment', 'No assessment available.')}
        </div>""", unsafe_allow_html=True)
        if extra_key and data.get(extra_key):
            st.markdown(f"*{data[extra_key]}*")
        recs = data.get("recommendations", [])
        if recs:
            st.markdown("**Recommendations:**")
            for r in recs: st.markdown(f"- {r}")

def highlight_fillers(text, filler_instances):
    """Highlight filler words in transcript text."""
    highlighted = text
    for fw in FILLER_WORDS:
        highlighted = highlighted.replace(
            f" {fw} ", f' <span class="filler-word">{fw}</span> '
        )
    return highlighted

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free at console.groq.com — powers both Whisper transcription AND AI feedback",
        placeholder="gsk_..."
    )
    if api_key and not api_key.strip().startswith("gsk_"):
        st.warning("⚠️ Groq keys start with **gsk_**")

    st.divider()
    st.markdown("""
    ### 🎯 Analysis Standard
    """)
    st.markdown("""
    <div style="background:#f0fff4;border:1px solid #9ae6b4;border-radius:8px;
                padding:.8rem 1rem;font-size:.82rem;color:#276749">
        ⚙️ <b>Fixed for accuracy:</b><br>
        • Every 4th frame sampled<br>
        • Up to 500 frames analysed<br>
        • Full audio transcription<br>
        • Consistent across all videos
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    ### 📋 Visual Detection
    - 👁️ Eye contact & gaze
    - 😊 Facial expressions
    - 🙆 Head pose
    - 🔄 Nodding detection
    - 📉 Gaze-away timestamps

    ### 🎙️ Audio Detection (NEW)
    - 📝 Full transcript
    - 🐌 Filler words (um, uh, like...)
    - ⏸️ Silence / pause windows
    - 💬 Speech rate (WPM)
    - 😰 Hesitation markers

    ### 🤖 AI Feedback
    - Cross-modal insights
    - Eye contact + speech correlation
    - Patient comfort analysis
    - Priority actions

    ### 🔑 Get Free Groq Key
    1. **console.groq.com**
    2. Sign up (email only)
    3. API Keys → Create Key
    """)
    st.divider()
    st.caption("Multi-Modal Communication Framework PoC")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Doctor-Patient Communication Analyzer</h1>
    <p>Multi-Modal Analysis · MediaPipe (Visual) + Whisper (Audio) + Groq Llama 3.3 70B (Feedback)</p>
</div>
""", unsafe_allow_html=True)

col_url, col_btn = st.columns([5, 1])
with col_url:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )
with col_btn:
    analyze_btn = st.button("▶ Analyze", use_container_width=True)

st.markdown("""
<div class="info-box">
💡 <b>Best results:</b> Use a front-facing video of a doctor with clear audio.
Try: <b>"OSCE clinical communication"</b> or <b>"doctor patient consultation skills"</b>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
if analyze_btn and youtube_url:
    if not api_key:
        st.error("⚠️ Enter your Groq API key in the sidebar. Free at console.groq.com")
        st.stop()

    tmp_dir = tempfile.mkdtemp()

    # STEP 1 — Download
    with st.status("📥 Downloading video…", expanded=True) as status:
        try:
            video_path, title, vid_duration = download_youtube(youtube_url, tmp_dir)
            status.update(label=f"✅ Downloaded: **{title}** ({int(vid_duration)}s)", state="complete")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    # STEP 2 — MediaPipe visual analysis
    prog_bar = st.progress(0, text="🔍 Analyzing frames with MediaPipe…")
    def update_prog(v):
        prog_bar.progress(v, text=f"🔍 Analyzing frames… {int(v*100)}%")
    with st.spinner(""):
        visual_stats = analyze_video(video_path, sample_rate=4, max_frames=500, progress_cb=update_prog)
    prog_bar.progress(1.0, text="✅ Visual analysis complete")
    time.sleep(0.2)
    prog_bar.empty()

    # STEP 3 — Audio extraction + Whisper transcription
    speech_stats   = None
    aligned_events = []
    audio_path     = None

    with st.status("🎙️ Extracting & transcribing audio with Whisper…", expanded=True) as audio_status:
        try:
            audio_path   = extract_audio(video_path, tmp_dir)
            transcript   = transcribe_audio(audio_path, api_key)
            speech_stats = analyze_speech(transcript)
            aligned_events = align_speech_with_face(speech_stats, visual_stats)
            audio_status.update(
                label=f"✅ Transcribed: **{speech_stats['word_count']} words** | "
                      f"**{speech_stats['speech_rate_wpm']} WPM** | "
                      f"**{speech_stats['filler_count']} filler words**",
                state="complete"
            )
        except Exception as e:
            audio_status.update(label=f"⚠️ Audio transcription failed: {e}", state="error")
            speech_stats = {
                "full_text": "", "word_count": 0, "speech_rate_wpm": 0,
                "filler_count": 0, "filler_instances": [], "silence_windows": [],
                "hesitation_count": 0, "segment_count": 0, "avg_segment_gap": 0,
                "segments": []
            }

    # STEP 4 — Groq multimodal feedback
    feedback = None
    with st.spinner("🤖 Generating multimodal AI feedback via Groq…"):
        try:
            feedback = get_llm_feedback(visual_stats, speech_stats, aligned_events, youtube_url, api_key)
        except ValueError as e:
            st.error(f"❌ {e}")
        except ConnectionError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"❌ Unexpected error: {type(e).__name__}: {e}")

    st.success(f"✅ Full multimodal analysis complete for **{title}**")
    st.divider()

    # ── TOP METRICS ──
    overall_score = feedback.get("overall_score", 0) if feedback else 0
    overall_grade = feedback.get("overall_grade", "N/A") if feedback else "N/A"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_metric("Overall Score", f"{overall_score}/100", overall_grade, score_class(overall_score))
    with c2:
        render_metric("Eye Contact", f"{visual_stats['eye_contact_pct']}%", "strong gaze",
                      score_class(int(visual_stats['eye_contact_pct'])))
    with c3:
        render_metric("Face Detected", f"{visual_stats['detection_rate']}%", "of frames",
                      score_class(int(visual_stats['detection_rate'])))
    with c4:
        render_metric("Speech Rate", f"{speech_stats['speech_rate_wpm']}", "words/min",
                      "score-good" if 120 <= speech_stats['speech_rate_wpm'] <= 160
                      else ("score-fair" if speech_stats['speech_rate_wpm'] > 0 else "score-poor"))
    with c5:
        render_metric("Filler Words", str(speech_stats['filler_count']), "detected",
                      "score-good" if speech_stats['filler_count'] < 5
                      else ("score-fair" if speech_stats['filler_count'] < 15 else "score-poor"))
    with c6:
        gc = len(visual_stats['gaze_away_windows'])
        render_metric("Gaze-Away", str(gc), "windows",
                      "score-good" if gc == 0 else ("score-fair" if gc < 5 else "score-poor"))

    st.markdown("<br>", unsafe_allow_html=True)

    # SUMMARY
    if feedback and feedback.get("summary"):
        st.markdown(f"""
        <div class="feedback-section">
            <h3>📋 Executive Summary</h3>
            <p style="color:#4a5568;line-height:1.7;font-size:.95rem">{feedback['summary']}</p>
        </div>""", unsafe_allow_html=True)

    # ── TRANSCRIPT + VISUAL SIDE BY SIDE ──
    t_col, v_col = st.columns([3, 2])

    with t_col:
        st.markdown("### 🎙️ Speech Analysis")
        if speech_stats["full_text"]:
            st.markdown('<span class="audio-badge">🎙️ Whisper Transcription</span>', unsafe_allow_html=True)
            highlighted = highlight_fillers(speech_stats["full_text"], speech_stats["filler_instances"])
            st.markdown(f'<div class="transcript-box">{highlighted}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">⚠️ No transcript available</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Speech metrics
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            wpm = speech_stats['speech_rate_wpm']
            note = "✅ Ideal" if 120 <= wpm <= 160 else ("⚠️ Too fast" if wpm > 160 else ("⚠️ Too slow" if wpm > 0 else "N/A"))
            st.metric("Speech Rate", f"{wpm} WPM", note)
        with sc2:
            st.metric("Filler Words", speech_stats['filler_count'], "lower is better")
        with sc3:
            st.metric("Pauses Detected", len(speech_stats['silence_windows']), "natural pauses ok")

        # Silence windows
        if speech_stats["silence_windows"]:
            st.markdown("#### ⏸️ Silence / Pause Windows")
            for sw in speech_stats["silence_windows"][:6]:
                st.markdown(f"""
                <div class="timeline-item" style="border-color:#d69e2e">
                    <div class="timeline-time" style="color:#d69e2e">⏱ {sw['start']}s – {sw['end']}s</div>
                    <div class="timeline-text">{sw['gap_s']}s silence — possible hesitation or thinking pause</div>
                </div>""", unsafe_allow_html=True)

    with v_col:
        st.markdown("### 👁️ Visual Analysis")
        total_expr = sum(visual_stats["expr_counts"].values())
        color_map  = {
            "Neutral"               : "#4299e1",
            "Attentive / Raised brow": "#38a169",
            "Speaking / Reacting"   : "#d69e2e",
            "Concerned / Furrowed"  : "#e53e3e",
            "No face detected"      : "#a0aec0",
        }
        for expr, cnt in sorted(visual_stats["expr_counts"].items(), key=lambda x: -x[1]):
            pct = round(cnt / max(total_expr, 1) * 100)
            render_progress(expr, pct, color_map.get(expr, "#718096"))

        if visual_stats["gaze_away_windows"]:
            st.markdown("#### ⚠️ Gaze-Away Moments")
            for s, e in visual_stats["gaze_away_windows"][:5]:
                st.markdown(f"""
                <div class="timeline-item">
                    <div class="timeline-time">⏱ {s}s – {e}s</div>
                    <div class="timeline-text">Reduced eye contact</div>
                </div>""", unsafe_allow_html=True)

        if visual_stats.get("nod_timestamps"):
            st.markdown("#### 🔄 Nodding Detected")
            nod_str = " · ".join(f"{t}s" for t in visual_stats["nod_timestamps"][:8])
            st.markdown(f'<div class="info-box">✅ Nodding at: {nod_str}</div>', unsafe_allow_html=True)

    st.divider()

    # ── ALIGNED TIMELINE ──
    if aligned_events:
        st.markdown("### 🔗 Multimodal Timeline (Speech + Face)")
        st.caption("What was said vs what the face was doing at the same moment")
        for ev in aligned_events[:10]:
            eye_color = "#38a169" if ev["eye_contact"] and ev["eye_contact"] > 0.45 else "#e53e3e"
            silence_badge = ' <span class="silence-tag">⏸ pause before</span>' if ev["silence_before"] else ""
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;
                        padding:.8rem 1rem;margin-bottom:.5rem;font-size:.85rem">
                <div style="color:#4299e1;font-weight:600;font-size:.75rem">{ev['time']}{silence_badge}</div>
                <div style="color:#2d3748;margin:.3rem 0;font-style:italic">"{ev['speech']}"</div>
                <div style="display:flex;gap:1rem;color:#718096;font-size:.78rem">
                    <span>👁️ Eye: <b style="color:{eye_color}">{ev['eye_contact']}</b></span>
                    <span>😊 {ev['expression']}</span>
                    <span>🙆 {ev['head_pose']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── DETAILED FEEDBACK ──
    st.markdown("### 🔬 Detailed Feedback")
    if feedback:
        # Multimodal insights first
        insights = feedback.get("multimodal_insights", [])
        if insights:
            st.markdown("#### 🔗 Cross-Modal Insights")
            for ins in insights:
                st.markdown(f"""
                <div style="background:#faf5ff;border-left:4px solid #9f7aea;border-radius:8px;
                            padding:.8rem 1rem;margin-bottom:.5rem;font-size:.88rem;color:#553c9a">
                    💡 {ins}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            render_feedback_block("Eye Contact",      "👁️", feedback.get("eye_contact_feedback", {}))
            render_feedback_block("Speech & Language","🎙️", feedback.get("speech_feedback", {}), "filler_word_note")
        with fc2:
            render_feedback_block("Body Language",    "🙆", feedback.get("body_language_feedback", {}))
            render_feedback_block("Patient Comfort",  "💙", feedback.get("patient_comfort_analysis", {}))

        # Priority actions
        st.markdown("### 🎯 Priority Actions")
        if feedback.get("priority_actions"):
            p1, p2, p3 = st.columns(3)
            rank_colors = ["#e53e3e", "#d69e2e", "#3182ce"]
            for action, col in zip(feedback["priority_actions"], [p1, p2, p3]):
                rank  = action.get("rank", 1)
                color = rank_colors[min(rank - 1, 2)]
                with col:
                    st.markdown(f"""
                    <div style="background:white;border-left:4px solid {color};border-radius:8px;
                                padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,.06);height:100%">
                        <div style="font-size:.75rem;font-weight:700;color:{color};text-transform:uppercase">
                            #{rank} Priority</div>
                        <div style="font-size:.9rem;font-weight:600;color:#2d3748;margin:.3rem 0">
                            {action.get('action','')}</div>
                        <div style="font-size:.8rem;color:#718096">{action.get('rationale','')}</div>
                    </div>""", unsafe_allow_html=True)

        # Strengths
        if feedback.get("strengths"):
            st.markdown("#### ✅ Strengths")
            for s in feedback["strengths"]:
                st.markdown(f"<span class='tag tag-green'>✓ {s}</span>", unsafe_allow_html=True)

        # Clinical tips
        tips = feedback.get("clinical_communication_tips", [])
        if tips:
            st.markdown("### 💡 Clinical Communication Tips")
            tip_icons = ["🗣️", "👂", "🤝", "💬", "🌡️"]
            cols = st.columns(len(tips))
            for i, (tip, col) in enumerate(zip(tips, cols)):
                with col:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#ebf8ff,#e6fffa);border-radius:10px;
                                padding:1rem;text-align:center;border:1px solid #bee3f8;height:100%">
                        <div style="font-size:1.5rem">{tip_icons[i % len(tip_icons)]}</div>
                        <div style="font-size:.85rem;color:#2c5282;margin-top:.5rem;line-height:1.5">{tip}</div>
                    </div>""", unsafe_allow_html=True)

    # RAW DATA
    with st.expander("🗃️ Raw Analysis Data (JSON)"):
        st.json({
            "video_url"    : youtube_url,
            "video_title"  : title,
            "visual_stats" : {k: v for k, v in visual_stats.items() if k != "frames_data"},
            "speech_stats" : {k: v for k, v in speech_stats.items() if k != "segments"},
            "aligned_events": aligned_events,
            "ai_feedback"  : feedback,
        })

    # Cleanup
    try:
        for f in [video_path, audio_path]:
            if f and os.path.exists(f): os.remove(f)
        os.rmdir(tmp_dir)
    except Exception:
        pass

elif analyze_btn and not youtube_url:
    st.warning("Please enter a YouTube URL to analyze.")

# FOOTER
st.markdown("""
<div style="text-align:center;color:#a0aec0;font-size:.8rem;margin-top:3rem;
            padding:1rem;border-top:1px solid #e2e8f0">
    Multi-Modal Communication Framework PoC · MediaPipe + Whisper + Groq (Llama 3.3 70B)
</div>""", unsafe_allow_html=True)