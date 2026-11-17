import os
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .early-blight { background-color: #FEF3C7; color: #92400E; }
    .late-blight { background-color: #FEE2E2; color: #991B1B; }
    .healthy { background-color: #D1FAE5; color: #065F46; }
</style>
""", unsafe_allow_html=True)

CLASS_LABELS = {
    "Potato___Early_blight": {
        "title": "Early Blight (Alternaria solani)",
        "badge_class": "early-blight",
        "description": "Early blight is caused by the fungus *Alternaria solani*. It produces small, dark brown-to-black spots with concentric rings (target board pattern) primarily on older foliage.",
        "treatment": [
            "Apply approved copper-based or chlorothalonil fungicides at initial symptoms.",
            "Maintain adequate plant spacing to enhance airflow and leaf drying.",
            "Water at the base of plants using drip irrigation to avoid wet leaves.",
            "Rotate crops with non-solanaceous species for at least 2–3 years."
        ]
    },
    "Potato___Late_blight": {
        "title": "Late Blight (Phytophthora infestans)",
        "badge_class": "late-blight",
        "description": "Late blight is caused by the oomycete *Phytophthora infestans*. It causes rapidly expanding dark, water-soaked lesions on leaves and stems, often surrounded by a pale green halo, and can destroy foliage in days.",
        "treatment": [
            "Apply specialized systemic fungicides (e.g., cymoxanil, mandipropamid, or mancozeb) immediately.",
            "Remove and safely destroy (burn or bury) infected haulms to prevent airborne spore spread.",
            "Avoid overhead irrigation during cool, humid periods.",
            "Plant certified disease-free seed tubers and resistant cultivars if available."
        ]
    },
    "Potato___healthy": {
        "title": "Healthy Potato Leaf",
        "badge_class": "healthy",
        "description": "The leaf appears vigorous, green, and free from pathogenic lesions or discoloration.",
        "treatment": [
            "Continue regular monitoring for pests (e.g., aphids, beetles).",
            "Maintain balanced soil nutrition and proper moisture levels.",
            "Practice proactive weed control and crop sanitation."
        ]
    }
}

MODEL_PATH = Path("potato_cnn_model.keras")
DEFAULT_CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file '{MODEL_PATH}' not found in current directory.")
        return None
    return tf.keras.models.load_model(str(MODEL_PATH))

def load_class_names():
    report_file = Path("classification_report.txt")
    if report_file.exists():
        lines = [line.strip() for line in report_file.read_text().splitlines() if line.strip()]
        if len(lines) == 3:
            return lines
    return DEFAULT_CLASSES

def predict_image(model, image: Image.Image, class_names):
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Target size: 128x128
    resized_img = image.resize((128, 128))
    img_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[predicted_idx])
    
    return predicted_class, confidence, predictions

# Sidebar
st.sidebar.title("🥔 Potato Health AI")
st.sidebar.markdown(
    """
    **Deep Learning Potato Disease Classifier**
    
    Detect and diagnose potato leaf diseases instantly using our trained Convolutional Neural Network (CNN).
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Navigation & Samples")

sample_option = st.sidebar.selectbox(
    "Test with sample images:",
    options=["Upload my own image", "Sample: leaf.jpg", "Sample from Early Blight", "Sample from Late Blight", "Sample from Healthy"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Architecture")
st.sidebar.info(
    """
    - **Backbone**: Custom 4-Stage CNN (32-64-128-256 filters) with BatchNorm & Dropout
    - **Input Size**: 128 × 128 pixels (RGB)
    - **Validation Accuracy**: ~98%
    - **Output Classes**: 3
    """
)

# Main Header
st.markdown('<div class="main-title">🥔 Potato Leaf Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a potato leaf photo to diagnose diseases and view actionable treatment guides.</div>', unsafe_allow_html=True)

model = load_trained_model()
class_names = load_class_names()

# Load image based on selection
input_image = None
image_source_name = ""

if sample_option == "Upload my own image":
    uploaded_file = st.file_uploader("Upload leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        image_source_name = uploaded_file.name
elif sample_option == "Sample: leaf.jpg":
    leaf_path = Path("leaf.jpg")
    if leaf_path.exists():
        input_image = Image.open(leaf_path)
        image_source_name = "leaf.jpg"
elif sample_option == "Sample from Early Blight":
    sample_dir = Path("data/Potato___Early_blight")
    if sample_dir.exists():
        files = list(sample_dir.glob("*.JPG")) + list(sample_dir.glob("*.jpg"))
        if files:
            input_image = Image.open(files[0])
            image_source_name = files[0].name
elif sample_option == "Sample from Late Blight":
    sample_dir = Path("data/Potato___Late_blight")
    if sample_dir.exists():
        files = list(sample_dir.glob("*.JPG")) + list(sample_dir.glob("*.jpg"))
        if files:
            input_image = Image.open(files[0])
            image_source_name = files[0].name
elif sample_option == "Sample from Healthy":
    sample_dir = Path("data/Potato___healthy")
    if sample_dir.exists():
        files = list(sample_dir.glob("*.JPG")) + list(sample_dir.glob("*.jpg"))
        if files:
            input_image = Image.open(files[0])
            image_source_name = files[0].name

# Display results layout
if input_image is not None and model is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown(f"### 📷 Input Image (`{image_source_name}`)")
        st.image(input_image, use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing leaf with CNN model..."):
            predicted_class, confidence, predictions = predict_image(model, input_image, class_names)
            info = CLASS_LABELS.get(predicted_class, {
                "title": predicted_class,
                "badge_class": "early-blight",
                "description": "No description available.",
                "treatment": []
            })
            
            st.markdown("### 🔍 Diagnosis Result")
            st.markdown(
                f"""
                <div class="prediction-card">
                    <span class="status-badge {info['badge_class']}">{info['title']}</span>
                    <h2 style="margin-top: 0.8rem; margin-bottom: 0.2rem;">{confidence * 100:.1f}% Confidence</h2>
                    <p style="color: #64748B; font-size: 0.95rem;">Model diagnosis based on leaf visual patterns</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### Probability Breakdown")
            for name, score in zip(class_names, predictions):
                label_info = CLASS_LABELS.get(name, {})
                display_name = label_info.get("title", name).split(" (")[0]
                st.write(f"**{display_name}**: `{score * 100:.2f}%`")
                st.progress(float(score))
    
    st.markdown("---")
    st.markdown("### 💡 Disease Information & Recommended Treatment")
    st.markdown(f"**Description**: {info['description']}")
    
    if info["treatment"]:
        st.markdown("**Management & Treatment Steps:**")
        for step in info["treatment"]:
            st.markdown(f"- {step}")

else:
    st.info("👆 Please upload an image or choose one of the sample images from the sidebar to start diagnosis.")
