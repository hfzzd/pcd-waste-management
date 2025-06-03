import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

INPUT_DIR = "data/preprocessed"
OUTPUT_CSV = "data/features/texture/texture_features.csv"

def extract_texture_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    return [contrast, homogeneity, energy, correlation]

def run():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    features = []
    for file in os.listdir(INPUT_DIR):
        if "_gray.jpg" in file:
            path = os.path.join(INPUT_DIR, file)
            feat = extract_texture_features(path)
            label = file.split("_")[0]
            features.append([file] + feat + [label])
    df = pd.DataFrame(features)
    df.to_csv(OUTPUT_CSV, index=False)
