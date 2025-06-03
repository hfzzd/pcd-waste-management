import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

COLOR = "data/features/color/color_features.csv"
SHAPE = "data/features/shape/shape_features.csv"
TEXTURE = "data/features/texture/texture_features.csv"

def train_and_evaluate(csv_path, feature_type):
    print(f"\n== Klasifikasi fitur {feature_type.upper()} ==")
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, X_train)
    y_pred_knn = knn.predict(X_test)
    print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")

    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")

def run():
    train_and_evaluate(COLOR, "warna")
    train_and_evaluate(SHAPE, "bentuk")
    train_and_evaluate(TEXTURE, "tekstur")
