import preprocessing
import classify 
import feature_extraction.color
import feature_extraction.shape
import feature_extraction.texture

def main():
    print("=== Preprocessing ===")
    preprocessing.run()

    print("=== Feature Extraction ===")
    feature_extraction.color.run()
    feature_extraction.shape.run()
    feature_extraction.texture.run()

    print("=== Klasifikasi ===")
    classify.run()

if __name__ == "__main__":
    main()
