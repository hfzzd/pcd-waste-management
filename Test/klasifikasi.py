import cv2
import numpy as np
import os
import joblib
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Keterangan label
# -------------------------
LABEL_DESCRIPTIONS = {
    'plastik': '♻️ Plastik: Limbah anorganik seperti botol, kantong, bungkus makanan.',
    'kertas': '📄 Kertas: Termasuk kertas koran, kardus, dan dokumen.',
    'organik': '🍂 Organik: Sisa makanan, daun, atau bahan mudah terurai.'
}

# -------------------------
# FEATURE EXTRACTOR CLASS
# -------------------------
class FeatureExtractor:
    def __init__(self):
        self.features = []
        self.labels = []

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
        return np.concatenate([h_hist, s_hist, v_hist, [h_mean, s_mean, v_mean]])

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
        circularity = (4 * np.pi * area) / (contour_length ** 2) if contour_length > 0 else 0
        eccentricity = np.sqrt(1 - (min(w, h) / max(w, h))**2) if max(w, h) != 0 else 0
        num_contours = len(contours)
        return np.concatenate([
            hu_moments,
            [aspect_ratio, solidity, contour_length, extent, circularity, eccentricity, num_contours]
        ])

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
        return np.concatenate([lbp_hist, [contrast, dissimilarity, homogeneity, energy, correlation]])

    def extract_all_features(self, img_path, label=None):
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Gagal membaca gambar: {img_path}")
            return None
        color_features = self.extract_color_features(img)
        shape_features = self.extract_shape_features(img)
        texture_features = self.extract_texture_features(img)
        all_features = np.concatenate([color_features, shape_features, texture_features])
        if label is not None:
            self.features.append(all_features)
            self.labels.append(label)
        return all_features

# -------------------------
# PELATIHAN MODEL
# -------------------------
def process_images_in_folder(folder_path, label, extractor):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            extractor.extract_all_features(path, label)

def train_and_save_model():
    extractor = FeatureExtractor()
    process_images_in_folder('Test/dataset_sampah/plastik', 'plastik', extractor)
    process_images_in_folder('Test/dataset_sampah/kertas', 'kertas', extractor)
    process_images_in_folder('Test/dataset_sampah/organik', 'organik', extractor)

    X = np.array(extractor.features)
    y = np.array(extractor.labels)
    if len(X) < 3:
        print("❌ Dataset terlalu kecil untuk melatih model (minimal 3 data).")
        return
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    joblib.dump(model, 'Test/model_knn.pkl')
    joblib.dump(le, 'Test/label_encoder.pkl')
    print("✅ Model dilatih dan disimpan!")

# -------------------------
# PREDIKSI SEMUA GAMBAR DALAM FOLDER
# -------------------------
def predict_all_images_in_folder(folder_path):
    if not os.path.exists('Test/model_knn.pkl') or not os.path.exists('Test/label_encoder.pkl'):
        print("❌ Model belum dilatih.")
        return

    model = joblib.load('Test/model_knn.pkl')
    le = joblib.load('Test/label_encoder.pkl')
    extractor = FeatureExtractor()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            features = extractor.extract_all_features(image_path)
            if features is None:
                continue
            features = features.reshape(1, -1)
            prediction = model.predict(features)
            label = le.inverse_transform(prediction)[0]
            description = LABEL_DESCRIPTIONS.get(label, 'Tidak ada deskripsi untuk label ini.')
            print(f"\n🔍 Gambar: {filename}")
            print(f"📌 Prediksi: {label.upper()}")
            print(f"🧾 Deskripsi: {description}")

# -------------------------
# MAIN PROGRAM
# -------------------------
if __name__ == "__main__":
    train_and_save_model()

    for category in ['plastik', 'kertas', 'organik']:
        folder = f'Test/dataset_sampah/{category}'
        print(f"\n🗂️ Memproses folder: {folder}")
        predict_all_images_in_folder(folder)
