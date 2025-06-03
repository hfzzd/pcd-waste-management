import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# === KONFIGURASI ===
MODEL_PATH = "models/svm_color.pkl"          # ubah sesuai fitur
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
IMAGE_PATH = "data/test_input.jpg"            # path gambar input

def extract_color_feature(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def run():
    print("=== Prediksi Sampah dari Gambar ===")
    feat = extract_color_feature(IMAGE_PATH)
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    
    prediction = model.predict([feat])
    label = le.inverse_transform(prediction)
    print(f"Prediksi Jenis Sampah: {label[0]}")

if __name__ == "__main__":
    run()
