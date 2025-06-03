from src import preprocessing, feature_extraction, classification

def main():
    print("=== Preprocessing ===")
    preprocessing.run()

    print("=== Feature Extraction ===")
    feature_extraction.color.run()
    feature_extraction.shape.run()
    feature_extraction.texture.run()

    print("=== Classification ===")
    classification.run()

if __name__ == "__main__":
    main()