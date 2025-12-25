# =======================
# Imports
# =======================
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Wildfire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #FF4B4B;
}
</style>
""", unsafe_allow_html=True)

# =======================
# Constants
# =======================
IMG_SIZE = 224
CLASS_LABELS = {0: "No Wildfire", 1: "Wildfire"}

# =======================
# Load Model
# =======================
MODEL_URL = "https://drive.google.com/uc?id=1PnOX7t7o2Qqly-3nqVlSLa5LoGApvGjW"
MODEL_PATH = "best_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    if model.output_shape[-1] != 1:
        raise ValueError("Model must be binary sigmoid")
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.sidebar.error(str(e))
    model = None

# =======================
# Preprocessing
# =======================
def preprocess_image(image, debug=False):
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = img_to_array(image)

    preprocessing_method = st.sidebar.selectbox(
        "🎯 Model Architecture",
        [
            "EfficientNetB4 (Default)",
            "ResNet50/ResNet101/ResNet152",
            "VGG16/VGG19",
            "MobileNetV2",
            "InceptionV3/Xception",
            "DenseNet"
        ],
        index=0
    )

    if preprocessing_method == "EfficientNetB4 (Default)":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "ResNet50/ResNet101/ResNet152":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "VGG16/VGG19":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "InceptionV3/Xception":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "DenseNet":
        from tensorflow.keras.applications.densenet import preprocess_input
        img = preprocess_input(img)

    return np.expand_dims(img, axis=0)

# =======================
# Prediction
# =======================
def predict(image_array):
    preds = model.predict(image_array, verbose=0)
    prob = float(preds[0][0])
    label = int(prob >= 0.5)
    confidence = prob if label == 1 else 1 - prob
    return label, confidence, prob

# =======================
# Sentinel-2 Fetch
# =======================
def fetch_sentinel2_image(lat, lon, client_id, client_secret):
    from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    bbox_to_dimensions,
    BBox,
    CRS,
    MimeType
    )
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret

    bbox = BBox(
        bbox=[lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02],
        crs=CRS.WGS84
    )
    size = bbox_to_dimensions(bbox, resolution=10)

    evalscript = """
    //VERSION=3
    function setup() {
        return { input: ["B04", "B03", "B02"], output: { bands: 3 } };
    }
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=("2024-01-01", "2025-12-31"),
                mosaicking_order="mostRecent"
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config
    )

    data = request.get_data()
    return Image.fromarray(data[0])

# =======================
# Sidebar
# =======================
st.sidebar.title("⚙️ Settings")
debug_mode = st.sidebar.checkbox("🐛 Debug Mode")
show_comparison = st.sidebar.checkbox("🖼️ Show Image Comparison")

# =======================
# Main UI
# =======================
st.markdown('<p class="main-header">🔥 Wildfire Detection System</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🛰️ Sentinel-2", "📤 Upload", "ℹ️ About"])

# =======================
# TAB 1 — Sentinel-2
# =======================
with tab1:
    client_id = st.text_input("Sentinel Hub Client ID", type="password")
    client_secret = st.text_input("Sentinel Hub Client Secret", type="password")

    lat_value = st.number_input("Latitude", value=21.1)
    lon_value = st.number_input("Longitude", value=79.0)

    if st.button("🔍 Fetch & Analyze", type="primary", use_container_width=True):
        if not model:
            st.error("❌ Model not loaded")
        elif not client_id or not client_secret:
            st.warning("⚠️ Missing credentials")
        else:
            try:
                image = fetch_sentinel2_image(lat_value, lon_value, client_id, client_secret)
                st.image(image, caption="Sentinel-2 Image", use_container_width=True)

                img_array = preprocess_image(image, debug=debug_mode)
                label, confidence, raw_prob = predict(img_array)

                if label == 1:
                    st.error(f"🔥 Wildfire Detected ({confidence*100:.2f}%)")
                else:
                    st.success(f"✅ No Wildfire ({confidence*100:.2f}%)")

            except Exception as e:
                st.error(str(e))

# =======================
# TAB 2 — Upload
# =======================
with tab2:
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded and model:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        img_array = preprocess_image(image)
        label, confidence, _ = predict(img_array)

        if label == 1:
            st.error(f"🔥 Wildfire Detected ({confidence*100:.2f}%)")
        else:
            st.success(f"✅ No Wildfire ({confidence*100:.2f}%)")

# =======================
# TAB 3 — About
# =======================
with tab3:
    st.markdown("""
    **Educational wildfire detection system using Sentinel-2 imagery and deep learning.**
    Not for emergency decision-making.
    """)

