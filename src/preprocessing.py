import os
import cv2

INPUT_DIR = "data/original"
OUTPUT_DIR = "data/preprocessed"

def load_and_resize(image_path, size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    return img

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def preprocess_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(INPUT_DIR, file)
            img = load_and_resize(path)
            gray = convert_to_grayscale(img)
            hsv = convert_to_hsv(img)
            filename = os.path.splitext(file)[0]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename}_rgb.jpg"), img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename}_gray.jpg"), gray)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{filename}_hsv.jpg"), hsv)

def run():
    preprocess_images()