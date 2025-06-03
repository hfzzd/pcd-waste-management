import cv2
import numpy as np
import os
import pandas as pd

INPUT_DIR = "data/preprocessed"
OUTPUT_CSV = "data/features/shape/shape_features.csv"

def extract_shape_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu_moments = cv2.HuMoments(cv2.moments(thresh)).flatten()
    return hu_moments

def run():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    features = []
    for file in os.listdir(INPUT_DIR):
        if "_gray.jpg" in file:
            path = os.path.join(INPUT_DIR, file)
            feat = extract_shape_features(path)
            label = file.split("_")[0]
            features.append([file] + list(feat) + [label])
    df = pd.DataFrame(features)
    df.to_csv(OUTPUT_CSV, index=False)