cat > README.md << 'EOF'
# Doctor-Patient Communication Analyzer

Multi-modal AI analysis of clinical consultations.

## Stack
- MediaPipe — facial landmark detection
- Whisper (Groq) — speech transcription  
- Llama 3.3 70B (Groq) — clinical feedback

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Version
v1 — Visual + Audio + Multimodal AI Feedback
EOF

git add README.md
git commit -m "Add README"
git push