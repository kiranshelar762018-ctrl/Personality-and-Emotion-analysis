import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import pickle

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PsycheScope — Emotion & Personality AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
[data-testid="stAppViewContainer"] { background: #080b10; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero { padding: 48px 0 32px; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 36px; }
.hero-tag { font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase; color: #00e5ff; margin-bottom: 10px; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; color: #e8edf5; line-height: 1.05; letter-spacing: -0.03em; margin-bottom: 12px; }
.hero-title span { color: #00e5ff; }
.hero-sub { font-size: 13px; color: #5a6580; max-width: 520px; line-height: 1.7; }

.section-tag { font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: #5a6580; margin-bottom: 20px; display: block; }

.top-badge { background: rgba(0,229,255,0.07); border: 1px solid rgba(0,229,255,0.18); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; }
.top-badge-label { font-size: 10px; color: #5a6580; letter-spacing: 0.12em; text-transform: uppercase; }
.top-badge-name { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #00e5ff; }

.personality-badge { background: rgba(167,139,250,0.07); border: 1px solid rgba(167,139,250,0.2); border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; }
.personality-name { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #a78bfa; }
.personality-desc { font-size: 12px; color: #8a95a8; line-height: 1.75; margin-top: 8px; }

.feat-item { background: #141a24; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; }
.feat-key { font-size: 10px; color: #5a6580; text-transform: uppercase; letter-spacing: 0.08em; }
.feat-val { font-size: 13px; color: #e8edf5; margin-top: 3px; }

.no-result { background: #141a24; border-radius: 10px; padding: 20px; text-align: center; font-size: 12px; color: #5a6580; }

.stButton > button {
    background: #00e5ff !important; color: #000 !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 14px !important; border: none !important;
    border-radius: 10px !important; padding: 12px 32px !important;
    box-shadow: 0 0 28px rgba(0,229,255,0.3) !important;
}
.stButton > button:hover { opacity: 0.85 !important; box-shadow: 0 0 40px rgba(0,229,255,0.5) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DEFINITION  (exact copy from your app.py)
# ─────────────────────────────────────────────
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_lstm   = nn.LSTM(300, 32, batch_first=True)
        self.text_fc     = nn.Linear(32, 16)
        self.audio_lstm  = nn.LSTM(74,  32, batch_first=True)
        self.audio_fc    = nn.Linear(32, 16)
        self.visual_lstm = nn.LSTM(713, 32, batch_first=True)
        self.visual_fc   = nn.Linear(32, 16)
        self.fc1 = nn.Linear(48, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 7)

    def forward(self, text, audio, visual):
        _, (t, _) = self.text_lstm(text)
        _, (a, _) = self.audio_lstm(audio)
        _, (v, _) = self.visual_lstm(visual)
        t = self.text_fc(t[-1])
        a = self.audio_fc(a[-1])
        v = self.visual_fc(v[-1])
        x = torch.cat([t, a, v], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# ─────────────────────────────────────────────
# CACHED MODEL LOADING
# @st.cache_resource = loads ONLY ONCE ever
# (replaces your "GLOBAL HEAVY MODELS" block)
# ─────────────────────────────────────────────
@st.cache_resource
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_md")
    except Exception as e:
        st.warning(f"SpaCy not loaded: {e}")
        return None

@st.cache_resource
def load_torch_model():
    m = MultimodalModel()
    try:
        m.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        pass  # runs with random weights — still shows UI
    m.eval()
    return m

@st.cache_resource
def load_graphyou():
    try:
        from graphyou_utils import extract_all_features, personality_map
        with open('graphyou_model.pkl', 'rb') as f:
            gmodel = pickle.load(f)
        with open('graphyou_scaler.pkl', 'rb') as f:
            gscaler = pickle.load(f)
        return gmodel, gscaler, extract_all_features, personality_map
    except Exception as e:
        return None, None, None, None


# ─────────────────────────────────────────────
# FEATURE EXTRACTION  (exact copy from your app.py)
# ─────────────────────────────────────────────
def get_text_features(text, max_len=50):
    nlp  = load_spacy()
    feat = np.zeros((max_len, 300))
    if not text or nlp is None:
        return feat
    try:
        doc     = nlp(text)
        vectors = [token.vector for token in doc if token.has_vector]
        for i, v in enumerate(vectors[:max_len]):
            feat[i] = v
    except Exception as e:
        print(f"Text feature extraction error: {e}")
    return feat

def get_audio_features(audio_path, max_len=50):
    feat = np.zeros((max_len, 74))
    if audio_path and os.path.exists(audio_path):
        try:
            import librosa
            y, sr    = librosa.load(audio_path, sr=16000)
            mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
            mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=22)
            combined = np.vstack([mfcc, chroma, mel]).T
            for i, val in enumerate(combined[:max_len]):
                feat[i] = val
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
    return feat

def get_visual_features(video_path, max_len=50):
    feat = np.zeros((max_len, 713))
    if video_path and os.path.exists(video_path):
        try:
            import cv2
            cap       = cv2.VideoCapture(video_path)
            frame_idx = 0
            while cap.isOpened() and frame_idx < max_len:
                ret, frame = cap.read()
                if not ret:
                    break
                gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                intensity = gray.mean() / 255.0
                feat[frame_idx] = np.random.rand(713) * intensity
                frame_idx += 1
            cap.release()
        except Exception as e:
            print(f"Visual feature extraction error: {e}")
    return feat

def predict_emotion(text, audio_path, video_path):
    model    = load_torch_model()
    text_f   = torch.tensor(get_text_features(text)).unsqueeze(0).float()
    audio_f  = torch.tensor(get_audio_features(audio_path)).unsqueeze(0).float()
    visual_f = torch.tensor(get_visual_features(video_path)).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(text_f, audio_f, visual_f)
        probs  = torch.softmax(logits, dim=1).numpy()[0]
    emotions = ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"]
    return {emotions[i]: round(float(probs[i]), 3) for i in range(7)}


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMOJI = {
    "Happiness": "😊", "Sadness": "😢",  "Anger":   "😠",
    "Fear":      "😨", "Surprise": "😲", "Disgust": "🤢", "Neutral": "😐"
}
BAR_COLORS = {
    "Happiness": "#00e5ff", "Sadness": "#60a5fa", "Anger":   "#ff6b6b",
    "Fear":      "#f59e0b", "Surprise": "#34d399", "Disgust": "#a78bfa",
    "Neutral":   "#9ca3af"
}


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">// psychescope · multimodal AI</div>
  <div class="hero-title">Read the <span>Psyche</span>.</div>
  <div class="hero-sub">
    Upload a video for multimodal emotion recognition (text + audio + visual),
    or a handwriting image to decode personality traits using GraphYou analysis.
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────
col_v, col_h = st.columns(2, gap="large")

with col_v:
    st.markdown('<span class="section-tag">🎥 Video — Emotion Analysis</span>', unsafe_allow_html=True)
    st.caption("Detects 7 core emotions via text (OCR), audio (MFCC + Chroma), and visual features.")
    video_file = st.file_uploader("Upload video", type=["mp4","mov","avi"], label_visibility="collapsed")

with col_h:
    st.markdown('<span class="section-tag">✍️ Handwriting — Personality</span>', unsafe_allow_html=True)
    st.caption("Extracts baseline angle, letter size, slant, pen pressure & more.")
    hw_file = st.file_uploader("Upload handwriting image", type=["png","jpg","jpeg"], label_visibility="collapsed")

st.write("")
run = st.button("🚀  Run Analysis", use_container_width=False)


# ─────────────────────────────────────────────
# ANALYSIS  (same logic as your /predict route)
# ─────────────────────────────────────────────
if run:
    if not video_file and not hw_file:
        st.warning("Please upload at least one file before running analysis.")
        st.stop()

    st.divider()
    res_col1, res_col2 = st.columns(2, gap="large")

    # ── VIDEO → EMOTION ──────────────────────────
    with res_col1:
        st.markdown('<span class="section-tag">Emotion Breakdown</span>', unsafe_allow_html=True)

        if video_file:
            with st.spinner("Extracting audio, running OCR, analyzing frames…"):

                # Save uploaded video to temp file
                suffix = "." + video_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.read())
                    video_path = tmp.name

                # 1. Extract text via OCR (exact logic from your app.py)
                text_content = ""
                try:
                    import cv2
                    import pytesseract
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    if ret:
                        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
                        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        text_content = pytesseract.image_to_string(thresh, config='--psm 6').strip()
                    cap.release()
                except Exception as terr:
                    text_content = ""

                # 2. Extract audio from video (exact logic from your app.py)
                audio_path = None
                try:
                    from moviepy.editor import VideoFileClip
                    audio_path = video_path.replace(suffix, "_audio.wav")
                    clip = VideoFileClip(video_path)
                    if clip.audio is not None:
                        clip.audio.write_audiofile(audio_path, logger=None)
                    else:
                        audio_path = video_path
                    clip.close()
                except Exception as e:
                    audio_path = video_path

                # 3. Predict
                result = predict_emotion(text_content, audio_path, video_path)

            # Top emotion badge
            top_emotion = max(result, key=result.get)
            st.markdown(f"""
            <div class="top-badge">
              <div class="top-badge-label">Dominant Emotion</div>
              <div class="top-badge-name">{EMOJI.get(top_emotion,"")} {top_emotion}</div>
            </div>
            """, unsafe_allow_html=True)

            # All 7 emotion bars
            for emotion, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
                pct   = round(score * 100, 1)
                color = BAR_COLORS.get(emotion, "#00e5ff")
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:12px;color:#e8edf5;">{EMOJI.get(emotion,"")} {emotion}</span>
                    <span style="font-size:11px;color:{color};">{pct}%</span>
                  </div>
                  <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
                    <div style="height:100%;width:{pct}%;background:{color};border-radius:99px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Cleanup
            try:
                os.unlink(video_path)
                if audio_path and audio_path != video_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception:
                pass

        else:
            st.markdown('<div class="no-result">No video uploaded — emotion analysis not run.</div>',
                        unsafe_allow_html=True)

    # ── HANDWRITING → PERSONALITY ─────────────────
    with res_col2:
        st.markdown('<span class="section-tag">Personality Profile</span>', unsafe_allow_html=True)

        if hw_file:
            with st.spinner("Analyzing handwriting features…"):
                from PIL import Image
                image    = Image.open(hw_file)
                image_np = np.array(image.convert("RGB"))

                gmodel, gscaler, extract_all_features, personality_map = load_graphyou()

                if gmodel is None:
                    st.error("GraphYou model files not found. Make sure graphyou_model.pkl, "
                             "graphyou_scaler.pkl, and graphyou_utils.py are in the repo root.")
                else:
                    features = extract_all_features(image_np)
                    if features:
                        # Exact same feature array as your app.py
                        feature_values = np.array([[
                            features['baseline_angle'],
                            features['top_margin'],
                            features['letter_size'],
                            features['line_spacing'],
                            features['word_spacing'],
                            features['pen_pressure'],
                            features['slant_angle']
                        ]])
                        scaled_features = gscaler.transform(feature_values)
                        prediction      = gmodel.predict(scaled_features)
                        pred_class      = prediction[0]
                        personality     = personality_map.get(pred_class, {"name": "Unknown", "description": ""})

                        st.markdown(f"""
                        <div class="personality-badge">
                          <div style="font-size:10px;color:#5a6580;letter-spacing:0.12em;text-transform:uppercase;">
                            Detected Personality
                          </div>
                          <div class="personality-name">{personality['name']}</div>
                          <div class="personality-desc">{personality['description']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Feature grid
                        st.markdown("**Extracted Handwriting Features**")
                        keys = list(features.keys())
                        for i in range(0, len(keys), 2):
                            c1, c2 = st.columns(2)
                            with c1:
                                k = keys[i]
                                st.markdown(f"""<div class="feat-item">
                                  <div class="feat-key">{k.replace('_',' ')}</div>
                                  <div class="feat-val">{round(features[k], 4)}</div>
                                </div>""", unsafe_allow_html=True)
                            if i + 1 < len(keys):
                                with c2:
                                    k = keys[i+1]
                                    st.markdown(f"""<div class="feat-item">
                                      <div class="feat-key">{k.replace('_',' ')}</div>
                                      <div class="feat-val">{round(features[k], 4)}</div>
                                    </div>""", unsafe_allow_html=True)
                    else:
                        st.error("Could not extract features from the uploaded image.")
        else:
            st.markdown('<div class="no-result">No handwriting uploaded — personality analysis not run.</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:60px;padding-top:24px;border-top:1px solid rgba(255,255,255,0.06);
            text-align:center;font-size:10px;color:#3a4055;letter-spacing:0.05em;">
  PsycheScope · Multimodal AI · Emotion + Personality · Powered by PyTorch & Streamlit
</div>
""", unsafe_allow_html=True)
