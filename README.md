# 🥔 Potato Leaf Disease Detection & Classification

A Deep Learning computer vision system designed to diagnose potato crop diseases from leaf imagery using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed as an interactive Streamlit web application.

---

## 📌 Project Overview

Potato crops are susceptible to fungal and fungal-like pathogens, predominantly **Early Blight** (*Alternaria solani*) and **Late Blight** (*Phytophthora infestans*), which can cause catastrophic crop yield losses if not identified and treated early.

This system provides:
1. **Automated Disease Classification**: Accurate diagnosis into three target classes.
2. **Interactive Web App**: Simple drag-and-drop image interface with probability breakdowns.
3. **Actionable Recommendations**: Treatment and disease management guidance upon detection.

---

## 🎯 Target Classes

| Class | Description | Primary Pathogen |
| :--- | :--- | :--- |
| **Early Blight** | Concentric target-like brown/black lesions on leaves | *Alternaria solani* |
| **Late Blight** | Rapidly expanding water-soaked dark lesions | *Phytophthora infestans* |
| **Healthy** | Vigorous green foliage without pathogenic lesions | None |

---

## 🧠 Model Architecture & Performance

- **Input Dimension**: $128 \times 128 \times 3$ (RGB)
- **Preprocessing & Augmentation**:
  - Rescaling ($1/255$)
  - Random flip (horizontal & vertical), zoom, and rotation
- **Backbone**:
  - 4 Sequential Conv2D stages ($32 \to 64 \to 128 \to 256$ filters, kernel size $3 \times 3$, `same` padding)
  - Batch Normalization (`momentum=0.9`) & ReLU activations
  - Max Pooling ($2 \times 2$)
- **Head**:
  - `GlobalAveragePooling2D`
  - Fully Connected Dense layer ($128$ units, ReLU)
  - Dropout ($0.30$) for regularization
  - Final Dense Softmax layer ($3$ classes)
- **Training Strategy**:
  - Adam optimizer ($\text{lr} = 0.001$)
  - Balanced loss weighting via square-root inverse frequency to counteract class imbalance
  - Callbacks: `EarlyStopping` (patience=8, restore best weights) and `ReduceLROnPlateau` (factor=0.5, patience=3)
- **Validation Accuracy**: **~98%**

---

## 📁 Project Directory Structure

```plaintext
├── app.py                      # Streamlit interactive web application
├── cnn_code.ipynb              # Main training, evaluation & metrics notebook
├── test.ipynb                  # Prototyping & early experimentation notebook
├── potato_cnn_model.keras      # Trained & serialized Keras model (~5.17 MB)
├── classification_report.txt   # Target class labels in index order
├── leaf.jpg                    # Sample test image
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── data/                       # Dataset directory (2,152 images)
    ├── Potato___Early_blight/  # 1,000 images
    ├── Potato___Late_blight/   # 1,000 images
    └── Potato___healthy/       # 152 images
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ recommended
- Recommended to use a virtual environment (`venv`)

### 2. Setup Virtual Environment

```bash
# Clone or navigate to the repository
cd gem_ai

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Running the Web Application

Launch the Streamlit app with:

```bash
streamlit run app.py
```

Once started, open your browser and go to:
- **Local URL**: `http://localhost:8501`

### Using the App:
1. **Upload**: Upload a photo of a potato leaf (`.jpg`, `.jpeg`, `.png`), or select one of the built-in sample images from the sidebar.
2. **Predict**: The CNN model analyzes the image in real time.
3. **Review**: View the diagnosis, confidence percentage, class distribution bar charts, and recommended agricultural treatments.

---

## 🔬 Training the Model

To retrain or modify the model architecture, open and execute [`cnn_code.ipynb`](cnn_code.ipynb) in Jupyter Notebook or VS Code:

```bash
jupyter notebook cnn_code.ipynb
```
