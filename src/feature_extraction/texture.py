import cv2
import numpy as np
import os
import pandas as pd
from mahotas import features

INPUT_DIR = "data/preprocessed"
OUTPUT_CSV = "data/features/texture/texture_features.csv"

def extract_texture_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    texture_features = features.haralick(img).mean(axis=0)
    return texture_features

def run():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    features_list = []
    for file in os.listdir(INPUT_DIR):
        if "_gray.jpg" in file:
            path = os.path.join(INPUT_DIR, file)
            feat = extract_texture_features(path)
            label = file.split("_")[0]
            features_list.append([file] + list(feat) + [label])
    df = pd.DataFrame(features_list)
    df.to_csv(OUTPUT_CSV, index=False)