# waste-classification
This project focuses on the classification of waste images using digital image processing techniques. The main components of the project include preprocessing of images, feature extraction (color, shape, and texture), and classification using machine learning algorithms.

## Project Structure
```
waste-classification
├── src
│   ├── main.py          # Entry point of the application
│   ├── preprocessing.py  # Image preprocessing functions
│   ├── feature_extraction
│   │   ├── color.py     # Color feature extraction
│   │   ├── shape.py     # Shape feature extraction
│   │   ├── texture.py   # Texture feature extraction
│   ├── classification.py # Model training and evaluation
│   └── utils.py         # Utility functions for file handling
├── data
│   ├── original/        # Directory for original images
│   ├── preprocessed/    # Directory for preprocessed images
│   └── features/
│       ├── color/      # Directory for color features
│       ├── shape/      # Directory for shape features
│       └── texture/     # Directory for texture features
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd waste-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your original waste images in the `data/original/` directory.

## Usage
To run the project, execute the following command:
```
python src/main.py
```

This will preprocess the images, extract features, and classify the waste images.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.