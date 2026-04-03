# Multimodal Personality & Emotion Analysis Dashboard

This project is a complete end-to-end Machine Learning AI web dashboard built with **Flask** and **PyTorch**. 
It uniquely combines three different AI modalities (Facial Expressions, Voice Tone, and OCR Handwriting Analysis) into a single overarching emotional prediction in real-time.

---

## 🚀 Quick Start Setup (For any Windows/Mac PC)

If you are running this project for the very first time on a new computer, follow these simple terminal commands to get all the heavy machine learning models installed:

### 1. Install Required Python Libraries
Open your Command Prompt or Terminal, navigate directly into this project folder (`cd path/to/Multi_Personality`), and run:
```bash
pip install -r requirements.txt
```

### 2. Download the Language Model
This project uses SpaCy for deep text vectorization. You must download its default English dictionary model so the backend doesn't crash:
```bash
python -m spacy download en_core_web_md
```

*(Note: EasyOCR's language modules will automatically download into your cache the very first time you upload a video).*

---

## 🧠 Running the Server

Once you have completed the installation above, you simply start the server like so:

```bash
python app.py
```

Wait until you see: `* Running on http://127.0.0.1:5000`

Then, open your web browser (Chrome, Edge, Firefox), type `127.0.0.1:5000` into the address bar, and start dragging and dropping MP4 videos into the dashboard!

---

### Included Helper Scripts:
* **`train_model.py`**: A synthetic training loop. Run `python train_model.py` to artificially re-generate the `model.pth` PyTorch neural weights file directly on your local CPU.
* **`download.py`**: A helper script utilizing `gdown` to pull large `.csd` or video samples directly from Google Drive.
* **`make_test_video.py`**: A custom `moviepy` tool designed to flawlessly fuse an image slide, a video MP4, and an audio OGG into a perfectly synced test video for Dashboard analysis. 
