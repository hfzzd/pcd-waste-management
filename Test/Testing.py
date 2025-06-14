import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import os
import pandas as pd
import keyboard  # Untuk mendeteksi tombol ESC

class FeatureExtractor:
    def __init__(self, output_dir='Test/hasil_ekstraksi'):
        self.features = []
        self.labels = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_color_features(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_img], [0], None, [8], [0, 180])
        s_hist = cv2.calcHist([hsv_img], [1], None, [8], [0, 256])
        v_hist = cv2.calcHist([hsv_img], [2], None, [8], [0, 256])
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        h_mean = np.mean(hsv_img[:, :, 0])
        s_mean = np.mean(hsv_img[:, :, 1])
        v_mean = np.mean(hsv_img[:, :, 2])
        color_features = np.concatenate([h_hist, s_hist, v_hist, [h_mean, s_mean, v_mean]])
        return color_features

    def extract_shape_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return np.zeros(15)

        cnt = max(contours, key=cv2.contourArea)
        hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        contour_length = cv2.arcLength(cnt, True)
        extent = float(area) / (w * h) if (w * h) > 0 else 0
        (x_circle, y_circle), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)
        circularity = (4 * np.pi * area) / (contour_length ** 2) if contour_length > 0 else 0
        eccentricity = np.sqrt(1 - (min(w, h) / max(w, h))**2) if max(w, h) != 0 else 0
        num_contours = len(contours)

        shape_features = np.concatenate([
            hu_moments,
            [aspect_ratio, solidity, contour_length, extent, circularity, eccentricity, num_contours]
        ])
        return shape_features

    def extract_texture_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        texture_features = np.concatenate([lbp_hist, [contrast, dissimilarity, homogeneity, energy, correlation]])
        return texture_features

    def extract_all_features(self, img_path, label=None):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Gagal memuat gambar: {img_path}")
            return None

        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        color_features = self.extract_color_features(img)
        shape_features = self.extract_shape_features(img)
        texture_features = self.extract_texture_features(img)
        all_features = np.concatenate([color_features, shape_features, texture_features])

        # Simpan hasil visualisasi
        self.show_preprocessing(img, f"Preprocessing - {label}", f"{base_filename}_preprocessing.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, 24, 3, method='uniform')
        self.show_lbp(gray, lbp, f"{base_filename}_lbp.jpg")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.show_hsv_visual(hsv, f"{base_filename}_hsv_visual.jpg")


        if label is not None:
            self.features.append(all_features)
            self.labels.append(label)

        return all_features

    def save_features_to_csv(self, output_file):
        if len(self.features) == 0:
            print("Tidak ada fitur yang tersimpan")
            return
        df = pd.DataFrame(self.features)
        df['label'] = self.labels
        df.to_csv(output_file, index=False)
        print(f"Fitur disimpan ke {output_file}")

    def show_preprocessing(self, img, title="Ekstraksi bentuk", filename="preprocessing.jpg"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = img.copy()
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(thresh, cmap='gray')
        plt.title("Threshold")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
        plt.title("Contours")
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()


    def show_lbp(self, gray, lbp, filename="lbp_texture.jpg"):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale")
        plt.subplot(1, 2, 2)
        plt.imshow(lbp, cmap='gray')
        plt.title("LBP Texture")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()


    def show_hsv_visual(self, hsv_img, filename="hsv_visual.jpg"):
        h, s, v = cv2.split(hsv_img)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(h, cmap='hsv')
        plt.title("Hue")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(s, cmap='gray')
        plt.title("Saturation")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(v, cmap='gray')
        plt.title("Value")
        plt.axis('off')

        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def process_images_in_folder(folder_path, label, extractor):
    for idx, filename in enumerate(os.listdir(folder_path)):
        if keyboard.is_pressed('esc'):
            print("⛔ Proses dihentikan oleh pengguna (ESC ditekan).")
            break
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            print(f"[{label}] Memproses gambar {idx + 1}: {filename}")
            extractor.extract_all_features(img_path, label)

if __name__ == "__main__":
    extractor = FeatureExtractor()
    process_images_in_folder('Test/dataset_sampah/plastik', 'plastik', extractor)
    process_images_in_folder('Test/dataset_sampah/kertas', 'kertas', extractor)
    process_images_in_folder('Test/dataset_sampah/organik', 'organik', extractor)
    extractor.save_features_to_csv('Test/hasil_ekstraksi_fitur.csv')
    print("✅ Ekstraksi fitur selesai!")
