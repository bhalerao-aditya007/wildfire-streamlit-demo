import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="Wildfire Detection", layout="wide")

st.title("🔥 Wildfire Detection Model Demo")
st.markdown("Upload an image to detect whether it contains a wildfire or not.")

# Sidebar for model path input
st.sidebar.header("Model Configuration")
MODEL_URL = "https://drive.google.com/uc?id=16NImzXCLMWDq3dJs6Wn0rl3EpywTItOd"
model_path = "phase1_best_model.keras"
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, model_path, quiet=False)

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"❌ Model not found at: {model_path}")
    st.info("Please enter a valid path to the directory containing 'best_model.keras'")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

try:
    model = load_model(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define class labels
class_labels = {0: "No Wildfire", 1: "Wildfire"}
IMG_SIZE = 224

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_wildfire(image_array):
    prediction = model.predict(image_array, verbose=0)
    confidence = prediction[0][0]
    class_id = 1 if confidence > 0.5 else 0
    return class_id, confidence

# Create two columns for upload methods
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        help="Supported formats: JPG, JPEG, PNG, BMP, GIF"
    )

with col2:
    st.subheader("📷 Or Use Camera")
    camera_image = st.camera_input("Take a picture")

# Process uploaded image or camera image
image_to_process = None
source_name = ""

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)
    source_name = uploaded_file.name
elif camera_image is not None:
    image_to_process = Image.open(camera_image)
    source_name = "Camera Capture"

if image_to_process is not None:
    # Display image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image_to_process, caption=f"Input: {source_name}", use_column_width=True)
    
    # Make prediction
    with col2:
        st.subheader("Prediction Results")
        
        img_array = preprocess_image(image_to_process)
        class_id, confidence = predict_wildfire(img_array)
        
        # Display results
        prediction_text = class_labels[class_id]
        confidence_pct = (confidence * 100) if class_id == 1 else ((1 - confidence) * 100)
        
        if class_id == 1:  # Wildfire detected
            st.error(f"🔥 **{prediction_text}**")
            st.metric("Confidence", f"{confidence_pct:.2f}%")
        else:  # No wildfire
            st.success(f"✅ **{prediction_text}**")
            st.metric("Confidence", f"{confidence_pct:.2f}%")
    
    # Visualization
    st.subheader("Confidence Score Breakdown")
    fig, ax = plt.subplots(figsize=(8, 4))
    
    classes = list(class_labels.values())
    scores = [1 - confidence, confidence]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(classes, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title('Prediction Confidence Score', fontsize=14, fontweight='bold')
    
    # Add percentage labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.markdown(
    """
    ### How to use:
    1. Enter the path to your trained model directory in the sidebar
    2. Upload an image or take a photo using your camera
    3. The model will predict whether the image contains a wildfire
    4. View the confidence score and prediction results
    """

)
