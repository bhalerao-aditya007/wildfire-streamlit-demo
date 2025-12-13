import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Wildfire Detection System",
    page_icon="🔥",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center;">🔥 Wildfire Detection System</h1>
    <p style="text-align:center; font-size:18px;">
    Upload a satellite or ground image to detect the presence of wildfire using a deep learning model.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =======================
# Model Download & Load
# =======================
MODEL_URL = "https://drive.google.com/uc?id=16NImzXCLMWDq3dJs6Wn0rl3EpywTItOd"
MODEL_PATH = "phase1_best_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model (one-time setup)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =======================
# Constants
# =======================
IMG_SIZE = 224
CLASS_LABELS = {0: "No Wildfire", 1: "Wildfire"}

# =======================
# Image Processing
# =======================
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image_array):
    prob = model.predict(image_array, verbose=0)[0][0]
    label = 1 if prob > 0.5 else 0
    confidence = prob if label == 1 else (1 - prob)
    return label, confidence, prob

# =======================
# Upload Section
# =======================
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =======================
# Prediction Section
# =======================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2], gap="large")

    # Display Image
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    with col2:
        st.subheader("🔍 Prediction Result")

        img_array = preprocess_image(image)
        label, confidence, raw_prob = predict(img_array)

        if label == 1:
            st.error("🔥 **Wildfire Detected**")
        else:
            st.success("✅ **No Wildfire Detected**")

        st.metric("Confidence", f"{confidence * 100:.2f}%")

        # Confidence Visualization
        fig, ax = plt.subplots(figsize=(6, 3))
        scores = [1 - raw_prob, raw_prob]
        classes = ["No Wildfire", "Wildfire"]

        bars = ax.bar(classes, scores)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score * 100:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        st.pyplot(fig)

# =======================
# Footer
# =======================
st.divider()
st.markdown(
    """
    **How it works**
    - Upload an image
    - Model analyzes visual wildfire patterns
    - Confidence score indicates prediction reliability
    """
)
