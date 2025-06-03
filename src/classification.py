from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

def train_and_evaluate(features_csv, feature_type):
    print(f"\n=== Classification using {feature_type.upper()} features ===")
    df = pd.read_csv(features_csv)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

    # Support Vector Machine Classifier
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")

def run():
    color_features_csv = "data/features/color/color_features.csv"
    shape_features_csv = "data/features/shape/shape_features.csv"
    texture_features_csv = "data/features/texture/texture_features.csv"
    
    train_and_evaluate(color_features_csv, "color")
    train_and_evaluate(shape_features_csv, "shape")
    train_and_evaluate(texture_features_csv, "texture")