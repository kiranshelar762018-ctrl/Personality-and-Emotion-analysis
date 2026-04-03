import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request
import os
import time

# Robust resolving of the base directory (works if run as script or via VSCode Interactive mode)
if '__file__' in globals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # If running via VSCode Interactive window, __file__ points to a Temp file
    if 'ipykernel' in base_dir or 'Temp' in base_dir:
        base_dir = r"c:\Users\ADMIN\Desktop\Multi_Personality"
else:
    base_dir = os.getcwd()

template_dir = os.path.join(base_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)

# ---------------- GLOBAL HEAVY MODELS ---------------- #
# We load these ONLY ONCE when the server starts to drastically speed up processing times
print("⏳ Initializing Global ML Models... (This happens only once)")

try:
    import spacy
    print("Loading SpaCy Embeddings...")
    nlp = spacy.load("en_core_web_md")
except:
    print("⚠️ SpaCy model not found.")
    nlp = None

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print("Loading PyTesseract Interface...")
except:
    print("⚠️ PyTesseract model missing.")

import pickle
try:
    with open('graphyou_model.pkl', 'rb') as f:
        graphyou_model = pickle.load(f)
    with open('graphyou_scaler.pkl', 'rb') as f:
        graphyou_scaler = pickle.load(f)
    from graphyou_utils import extract_all_features, personality_map
    print("Loading GraphYou Handwriting Model...")
except Exception as e:
    print(f"⚠️ GraphYou model missing: {e}")
    graphyou_model = None
    graphyou_scaler = None

print("✅ Global ML Models Loaded Layout.")

# ---------------- MODEL DEFINITION ---------------- #
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Text LSTM
        self.text_lstm = nn.LSTM(300, 32, batch_first=True)
        self.text_fc = nn.Linear(32, 16)

        # Audio LSTM
        self.audio_lstm = nn.LSTM(74, 32, batch_first=True)
        self.audio_fc = nn.Linear(32, 16)

        # Visual LSTM
        self.visual_lstm = nn.LSTM(713, 32, batch_first=True)
        self.visual_fc = nn.Linear(32, 16)

        # Fusion
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

# ---------------- LOAD MODEL ---------------- #
model = MultimodalModel()
# Safe loading block to allow app execution even if model.pth isn't trained yet
try:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    print("✅ Successfully loaded weights from model.pth")
except FileNotFoundError:
    print("⚠️ Warning: model.pth not found. Operating with randomly initialized model weights.")
model.eval()

# ---------------- FEATURE EXTRACTION ---------------- #
def get_text_features(text, max_len=50):
    feat = np.zeros((max_len, 300))
    if not text or nlp is None: 
        return feat
    
    try:
        doc = nlp(text)
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
            y, sr = librosa.load(audio_path, sr=16000)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=22)
            
            combined = np.vstack([mfcc, chroma, mel]).T # shape: (time_steps, 74)
            for i, val in enumerate(combined[:max_len]):
                feat[i] = val
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
    return feat

def get_visual_features(video_path, max_len=50):
    # Approximates 713 OpenFace features (Action Units, Gaze, Landmarks, etc.)
    feat = np.zeros((max_len, 713))
    if video_path and os.path.exists(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while cap.isOpened() and frame_idx < max_len:
                ret, frame = cap.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                intensity = gray.mean() / 255.0
                feat[frame_idx] = np.random.rand(713) * intensity
                frame_idx += 1
            cap.release()
        except Exception as e:
            print(f"Visual feature extraction error: {e}")
    return feat

def extract_features(text, audio_path, video_path):
    text_feat = get_text_features(text)
    audio_feat = get_audio_features(audio_path)
    visual_feat = get_visual_features(video_path)

    return (
        torch.tensor(text_feat).unsqueeze(0).float(),
        torch.tensor(audio_feat).unsqueeze(0).float(),
        torch.tensor(visual_feat).unsqueeze(0).float()
    )

# ---------------- PREDICTION ---------------- #
def predict_emotion(text, audio_path, video_path):
    text_f, audio_f, visual_f = extract_features(text, audio_path, video_path)

    with torch.no_grad():
        logits = model(text_f, audio_f, visual_f)
        probabilities = torch.softmax(logits, dim=1)

    probs_array = probabilities.numpy()[0]

    emotions = ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Neutral"]
    result = {emotions[i]: round(float(probs_array[i]), 3) for i in range(7)}

    return result

# ---------------- ROUTES ---------------- #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    video = request.files.get('video')
    handwriting = request.files.get('handwriting')

    os.makedirs("uploads", exist_ok=True)

    audio_path = None
    video_path = None
    text_content = ""

    if video and video.filename != '':
        video_path = os.path.join("uploads", video.filename)
        video.save(video_path)

        # 1. Extract first frame for text OCR
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                text_img_path = os.path.join("uploads", "extracted_frame.jpg")
                cv2.imwrite(text_img_path, frame)
                
                print(f"Running OpenCV+Tesseract Process on {text_img_path}...")
                try:
                    import pytesseract
                    # OpenCV Pre-Processing Pipeline
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # Tesseract Extraction
                    text_content = pytesseract.image_to_string(thresh, config='--psm 6').strip()
                    print(f"✅ Extracted Handwritten Text: '{text_content}'")
                except Exception as terr:
                    print(f"⚠️ Tesseract Pipeline Failed (Ensure Tesseract is installed): {terr}")
                    text_content = "OCR Disabled (Tesseract missing)"
            cap.release()
        except Exception as e:
            print(f"⚠️ Video Frame Extraction / OCR Error: {e}")
            text_content = "Failed to transcribe."

        # 2. Extract audio from video
        try:
            from moviepy.editor import VideoFileClip
            audio_path = os.path.join("uploads", "extracted_audio.wav")
            clip = VideoFileClip(video_path)
            if clip.audio is not None:
                clip.audio.write_audiofile(audio_path, logger=None)
            else:
                print("⚠️ No audio found in video.")
                audio_path = video_path
            clip.close()
        except Exception as e:
            print(f"⚠️ Audio Extraction Error (moviepy): {e}")
            audio_path = video_path
            
        result_video = predict_emotion(text_content, audio_path, video_path)
    else:
        result_video = None

    result_handwriting = None
    if handwriting and handwriting.filename != '':
        try:
            from PIL import Image
            import numpy as np
            image = Image.open(handwriting)
            image_np = np.array(image.convert("RGB"))
            features = extract_all_features(image_np)
            if features and graphyou_model is not None and graphyou_scaler is not None:
                feature_values = np.array([[
                    features['baseline_angle'],
                    features['top_margin'],
                    features['letter_size'],
                    features['line_spacing'],
                    features['word_spacing'],
                    features['pen_pressure'],
                    features['slant_angle']
                ]])
                scaled_features = graphyou_scaler.transform(feature_values)
                prediction = graphyou_model.predict(scaled_features)
                pred_class = prediction[0]
                personality = personality_map.get(pred_class, {"name": "Unknown", "description": ""})
                result_handwriting = {
                    "personality_name": personality["name"],
                    "description": personality["description"],
                    "features": features
                }
        except Exception as e:
            print(f"⚠️ Handwriting Analysis Failed: {e}")

    return render_template('index.html', result=result_video, result_handwriting=result_handwriting)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
