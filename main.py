from src import preprocessing, extract_color, extract_shape, extract_texture, classify

def main():
    print("=== Preprocessing ===")
    preprocessing.run()

    print("=== Ekstraksi Warna ===")
    extract_color.run()

    print("=== Ekstraksi Bentuk ===")
    extract_shape.run()

    print("=== Ekstraksi Tekstur ===")
    extract_texture.run()

    print("=== Klasifikasi (KNN dan SVM) ===")
    classify.run()

if __name__ == "__main__":
    main()
