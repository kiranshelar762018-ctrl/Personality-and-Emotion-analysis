import cv2
import numpy as np
from PIL import Image

# --- Personality Mapping ---
personality_map = {
    0: {"name": "Introverted & Thoughtful", "description": "Prefers solitude, enjoys deep thinking, and is highly reflective."},
    1: {"name": "Outgoing & Confident", "description": "Sociable, enjoys engaging with others, and exudes confidence."},
    2: {"name": "Creative & Expressive", "description": "Imaginative, highly expressive, and values artistic expression."},
    3: {"name": "Analytical & Detail-Oriented", "description": "Focused on precision, prefers logic over emotions."},
    4: {"name": "Empathetic & Compassionate", "description": "Emotionally attuned, values deep connections, and is highly empathetic."}
}

# --- Preprocessing ---
def preprocess_image(uploaded_file):
    try:
        if isinstance(uploaded_file, np.ndarray):
            img = uploaded_file
        else:
            image = Image.open(uploaded_file).convert("RGB")
            img = np.array(image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            pass
        else:
            raise ValueError("Unsupported image format.")

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

        if len(thresh.shape) != 2:
            raise ValueError("Preprocessed image is not 2D.")

        return thresh

    except Exception as e:
        print(f"Error while processing the image: {str(e)}")
        return None

# --- Baseline Angle ---
def estimate_baseline_angle(thresh):
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
    return float(np.mean(angles)) if angles else 0.0

# --- Top Margin ---
def estimate_top_margin(thresh):
    rows = np.sum(thresh, axis=1)
    top_margin = int(np.argmax(rows > 0))
    return float(top_margin)

# --- Letter Size ---
def estimate_letter_size(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours]
    return float(np.mean(heights)) if heights else 0.0

# --- Line Spacing ---
def estimate_line_spacing(thresh):
    rows = np.sum(thresh, axis=1)
    line_indices = np.where(rows > 0)[0]
    spacings = np.diff(line_indices)
    return float(np.mean(spacings)) if len(spacings) > 1 else 0.0

# --- Word Spacing ---
def estimate_word_spacing(thresh):
    cols = np.sum(thresh, axis=0)
    word_indices = np.where(cols > 0)[0]
    spacings = np.diff(word_indices)
    return float(np.mean(spacings)) if len(spacings) > 1 else 0.0

# --- Pen Pressure ---
def estimate_pen_pressure(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    return float(np.mean(gray))

# --- Slant Angle ---
def estimate_slant_angle(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        if len(c) >= 5:
            _, _, angle = cv2.fitEllipse(c)
            angles.append(angle)
    return float(np.mean(angles)) if angles else 0.0

# --- Feature Extraction ---
def extract_all_features(image):
    processed_img = preprocess_image(image)
    if processed_img is None:
        return None
    features = {
        'baseline_angle': estimate_baseline_angle(processed_img),
        'top_margin': estimate_top_margin(processed_img),
        'letter_size': estimate_letter_size(processed_img),
        'line_spacing': estimate_line_spacing(processed_img),
        'word_spacing': estimate_word_spacing(processed_img),
        'pen_pressure': estimate_pen_pressure(image),
        'slant_angle': estimate_slant_angle(processed_img)
    }
    for key, val in features.items():
        if np.isnan(val) or np.isinf(val):
            features[key] = 0.0
    return features
