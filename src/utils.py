import os
import pandas as pd
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            images.append((filename, img_path))
    return images

def save_features_to_csv(features, output_path):
    df = pd.DataFrame(features)
    df.to_csv(output_path, index=False)

def load_features_from_csv(input_path):
    return pd.read_csv(input_path)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)