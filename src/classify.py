import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

COLOR = "data/features/color/color_features.csv"
SHAPE = "data/features/shape/shape_features.csv"
TEXTURE = "data/features/texture/texture_features.csv"

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, le, df):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📊 Evaluasi Model: {model_name}")
    print(f"Akurasi  : {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall   : {rec:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print("\nClassification Report:")
    print(f"Total data: {len(df)}")
    print(f"Jumlah kelas: {len(le.classes_)}")
    print(f"Test size (jumlah data): {int(0.3 * len(df))}")

    print(classification_report(y_test, y_pred, zero_division=0))

    return y_pred

def train_and_evaluate(csv_path, feature_type):
    print(f"\n=== Klasifikasi fitur {feature_type.upper()} ===")
    try:
        df = pd.read_csv(csv_path)
        if df.shape[0] < 2:
            print("❌ Data terlalu sedikit.")
            return

        print("📊 Data Preview:")
        print(df.head())
        print("Distribusi label awal:")
        print(df.iloc[:, -1].value_counts())

        # Filter label yang hanya punya < 2 data
        df = df.groupby(df.columns[-1]).filter(lambda x: len(x) >= 2)

        print("\n✅ Distribusi label setelah difilter:")
        print(df.iloc[:, -1].value_counts())

        # Ambil fitur dan label
        # Asumsi kolom pertama adalah ID, kolom terakhir adalah label
        if df.shape[1] < 3:
            print("❌ Data tidak memiliki fitur yang cukup (minimal 1 fitur dan 1 label).")
            return


        X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")


        if X.isnull().values.any():
            print("❌ Terdapat nilai NaN di fitur.")
            return

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Stratified split hanya aman setelah filter
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        # KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        y_pred_knn = evaluate_model(knn, X_train, X_test, y_train, y_test, "KNN", le, df)

        # SVM
        svm = SVC(kernel='linear')
        y_pred_svm = evaluate_model(svm, X_train, X_test, y_train, y_test, "SVM", le, df)

        # Confusion Matrix SVM
        cm = confusion_matrix(y_test, y_pred_svm)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"Confusion Matrix - SVM ({feature_type})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {csv_path}")
    except Exception as e:
        print(f"❌ ERROR saat klasifikasi fitur {feature_type}: {e}")

def run():
    train_and_evaluate(COLOR, "warna")
    train_and_evaluate(SHAPE, "bentuk")
    train_and_evaluate(TEXTURE, "tekstur")
    print("\n=== Selesai ===")

if __name__ == "__main__":
    run()
