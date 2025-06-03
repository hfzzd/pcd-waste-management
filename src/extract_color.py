import cv2
import numpy as np
import os
import pandas as pd

INPUT_DIR = "data/preprocessed"
OUTPUT_CSV = "data/features/color/color_features.csv"

def extract_color_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def run():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    features = []
    for file in os.listdir(INPUT_DIR):
        if "_rgb.jpg" in file:
            path = os.path.join(INPUT_DIR, file)
            feat = extract_color_features(path)
            label = file.split("_")[0]  # asumsi nama file: plastik_01_rgb.jpg
            features.append([file] + list(feat) + [label])
    df = pd.DataFrame(features)
    df.to_csv(OUTPUT_CSV, index=False)
