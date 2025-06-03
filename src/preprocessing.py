import os
import cv2

INPUT_DIR = "data/original"
OUTPUT_DIR = "data/preprocessed"

def convert_and_resize(image_path, size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, gray, hsv

def run():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(INPUT_DIR, file)
            img, gray, hsv = convert_and_resize(path)
            filename = os.path.splitext(file)[0]
            cv2.imwrite(f"{OUTPUT_DIR}/{filename}_rgb.jpg", img)
            cv2.imwrite(f"{OUTPUT_DIR}/{filename}_gray.jpg", gray)
            cv2.imwrite(f"{OUTPUT_DIR}/{filename}_hsv.jpg", hsv)
