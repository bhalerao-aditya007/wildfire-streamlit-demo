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
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    bbox_to_dimensions,
    BBox,
    CRS,
    MimeType
)

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Wildfire Detection System",
    page_icon="🔥",
    layout="wide"
)

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
        with st.spinner("📥 Downloading model from Google Drive (one-time)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)

    # sanity check
    if model.output_shape[-1] != 1:
        raise ValueError("Model must be binary sigmoid with output shape (*, 1)")

    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model = None


# =======================
# Image Preprocessing
# =======================
def preprocess_image(image):
    if image is None:
        return None

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.asarray(image).astype("float32") / 255.0

    if img.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError("Invalid image shape")

    return np.expand_dims(img, axis=0)

# =======================
# Prediction
# =======================
def predict(image_array):
    preds = model.predict(image_array, verbose=0)

    if preds.ndim != 2 or preds.shape[1] != 1:
        raise ValueError("Model output must be (batch, 1) sigmoid")

    prob = float(preds[0][0])
    label = int(prob >= 0.5)
    confidence = prob if label == 1 else 1 - prob

    return label, confidence, prob

# =======================
# Sentinel-2 Fetch (CORRECT WAY)
# =======================
def fetch_sentinel2_image(lat, lon, client_id, client_secret):
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret

    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError("Missing Sentinel Hub credentials")

    bbox = BBox(
        bbox=[lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02],
        crs=CRS.WGS84
    )

    size = bbox_to_dimensions(bbox, resolution=10)

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
        };
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
        responses=[
            SentinelHubRequest.output_response("default", MimeType.PNG)
        ],
        bbox=bbox,
        size=size,
        config=config
    )

    data = request.get_data()
    return Image.fromarray(data[0])

# =======================
# Confidence Chart
# =======================
def create_confidence_chart(raw_prob):
    fig, ax = plt.subplots(figsize=(7, 3))
    scores = [1 - raw_prob, raw_prob]
    labels = ["No Wildfire", "Wildfire"]

    ax.barh(labels, scores)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Model Confidence")

    for i, v in enumerate(scores):
        ax.text(v + 0.02, i, f"{v*100:.1f}%", va="center")

    plt.tight_layout()
    return fig

# =======================
# UI
# =======================
st.title("🔥 Wildfire Detection System")

tab1, tab2 = st.tabs(["🛰 Sentinel-2", "📤 Upload Image"])

# =======================
# TAB 1 — Sentinel-2
# =======================
with tab1:
    st.subheader("Sentinel-2 Satellite Analysis")

    client_id = st.text_input("Sentinel Hub Client ID", type="password")
    client_secret = st.text_input("Sentinel Hub Client Secret", type="password")

    lat = st.number_input("Latitude", value=21.1, step=0.1)
    lon = st.number_input("Longitude", value=79.0, step=0.1)

    if st.button("Fetch & Analyze"):
        if not model:
            st.error("Model not loaded")
        else:
            try:
                with st.spinner("Fetching Sentinel-2 image..."):
                    image = fetch_sentinel2_image(lat, lon, client_id, client_secret)

                st.image(image, caption="Sentinel-2 True Color", use_column_width=True)

                img_array = preprocess_image(image)
                label, confidence, raw_prob = predict(img_array)

                if label == 1:
                    st.error(f"🔥 Wildfire Detected ({confidence*100:.1f}%)")
                else:
                    st.success(f"✅ No Wildfire Detected ({confidence*100:.1f}%)")

                fig = create_confidence_chart(raw_prob)
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(str(e))

# =======================
# TAB 2 — Upload
# =======================
with tab2:
    st.subheader("Upload Image")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded and model:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)

        img_array = preprocess_image(image)
        label, confidence, raw_prob = predict(img_array)

        if label == 1:
            st.error(f"🔥 Wildfire Detected ({confidence*100:.1f}%)")
        else:
            st.success(f"✅ No Wildfire Detected ({confidence*100:.1f}%)")

        fig = create_confidence_chart(raw_prob)
        st.pyplot(fig)
        plt.close()

# =======================
# Footer
# =======================
st.markdown("---")
st.caption(" Cloud presence depends on acquisition date • Sentinel-2 L2A • Educational use only")

