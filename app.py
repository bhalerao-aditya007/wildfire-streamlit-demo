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
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #FF4B4B;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
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
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    if model.output_shape[-1] != 1:
        raise ValueError("Model must be binary sigmoid")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model = None

# =======================
# Preprocessing
# =======================
def preprocess_image(image):
    image_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = img_to_array(image_resized)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# =======================
# Prediction with Full Output
# =======================
def predict(image_array):
    """Make prediction and return all model outputs"""
    preds = model.predict(image_array, verbose=0)
    prob = float(preds[0][0])
    label = int(prob >= 0.5)
    confidence = prob if label == 1 else 1 - prob
    
    # Additional model statistics
    stats = {
        'raw_probability': prob,
        'wildfire_probability': prob * 100,
        'no_wildfire_probability': (1 - prob) * 100,
        'confidence': confidence * 100,
        'prediction': CLASS_LABELS[label],
        'threshold': 0.5,
        'risk_level': get_risk_level(prob)
    }
    
    return label, confidence, stats

def get_risk_level(prob):
    """Determine risk level based on probability"""
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MODERATE"
    else:
        return "HIGH"

# =======================
# Visualization: Thermometer
# =======================
def create_thermometer(prob, label):
    """Create a thermometer-style visualization"""
    fig, ax = plt.subplots(figsize=(3, 8), facecolor='white')
    
    # Thermometer body
    ax.barh(0, 1, height=0.3, color='#e0e0e0', edgecolor='black', linewidth=2)
    
    # Color based on risk
    if prob < 0.3:
        color = '#2ecc71'
    elif prob < 0.7:
        color = '#f39c12'
    else:
        color = '#e74c3c'
    
    # Fill thermometer
    ax.barh(0, prob, height=0.3, color=color, edgecolor='black', linewidth=2)
    
    # Add percentage text
    ax.text(0.5, 0, f'{prob*100:.1f}%', ha='center', va='center', 
            fontsize=24, fontweight='bold', color='white')
    
    # Labels
    ax.text(-0.05, 0, '0%', ha='right', va='center', fontsize=12)
    ax.text(1.05, 0, '100%', ha='left', va='center', fontsize=12)
    
    # Title
    result_text = "🔥 WILDFIRE" if label == 1 else "✅ NO FIRE"
    ax.text(0.5, 0.5, result_text, ha='center', va='bottom', 
            fontsize=18, fontweight='bold', color=color)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.3, 0.6)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# =======================
# Image Comparison
# =======================
def show_image_comparison(original, processed):
    """Show side-by-side comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(original)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    processed_display = processed[0]
    processed_display = (processed_display - processed_display.min()) / (processed_display.max() - processed_display.min())
    ax2.imshow(processed_display)
    ax2.set_title("Processed (Model Input)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# =======================
# Sentinel-2 Fetch
# =======================
@st.cache_data(ttl=3600)
def fetch_sentinel2_image(lat, lon, client_id, client_secret):
    """Fetch Sentinel-2 image with caching"""
    from sentinelhub import (
        SHConfig, SentinelHubRequest, DataCollection,
        bbox_to_dimensions, BBox, CRS, MimeType
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
        return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
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
# Display Results
# =======================
def display_results(image, label, confidence, stats, show_comparison_flag):
    """Display prediction results with all parameters"""
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        # Thermometer visualization
        fig = create_thermometer(stats['raw_probability'], label)
        st.pyplot(fig)
        plt.close()
    
    with col3:
        # Result card
        if label == 1:
            st.error(f"### 🔥 WILDFIRE DETECTED")
        else:
            st.success(f"### ✅ NO WILDFIRE DETECTED")
        
        # Display all model parameters
        st.markdown("#### 📊 Model Output Parameters")
        st.metric("Wildfire Probability", f"{stats['wildfire_probability']:.2f}%")
        st.metric("No Wildfire Probability", f"{stats['no_wildfire_probability']:.2f}%")
        st.metric("Confidence Score", f"{stats['confidence']:.2f}%")
        st.metric("Risk Level", stats['risk_level'])
        st.metric("Decision Threshold", f"{stats['threshold']:.2f}")
    
    # Image comparison
    if show_comparison_flag:
        st.markdown("---")
        st.subheader("🔍 Image Processing Comparison")
        img_array = preprocess_image(image)
        fig_comp = show_image_comparison(image, img_array)
        st.pyplot(fig_comp)
        plt.close()

# =======================
# Main UI
# =======================
st.markdown('<p class="main-header">🔥 Wildfire Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered wildfire detection using satellite imagery</p>', unsafe_allow_html=True)

# Settings in expander
with st.expander("⚙️ Settings"):
    show_comparison = st.checkbox("Show Image Comparison", value=False)

tab1, tab2, tab3 = st.tabs(["🛰️ Satellite Analysis", "📤 Upload Image", "ℹ️ About"])

# =======================
# TAB 1 — Sentinel-2
# =======================
with tab1:
    st.subheader("Real-time Sentinel-2 Satellite Analysis")
    
    # Credentials
    col1, col2 = st.columns(2)
    with col1:
        client_id = st.text_input("Sentinel Hub Client ID", type="password",
                                   help="Get free credentials at https://www.sentinel-hub.com/")
    with col2:
        client_secret = st.text_input("Sentinel Hub Client Secret", type="password")
    
    st.markdown("---")
    st.markdown("### 📍 Enter Location Coordinates")
    
    # Coordinates with N/S/E/W
    col1, col2, col3, col4 = st.columns([3, 1, 3, 1])
    
    with col1:
        lat_value = st.number_input(
            "Latitude",
            value=34.0,
            min_value=0.0,
            max_value=90.0,
            step=0.1,
            format="%.4f"
        )
    
    with col2:
        lat_dir = st.selectbox("", ["N", "S"], key="lat_dir")
    
    with col3:
        lon_value = st.number_input(
            "Longitude",
            value=118.2,
            min_value=0.0,
            max_value=180.0,
            step=0.1,
            format="%.4f"
        )
    
    with col4:
        lon_dir = st.selectbox(" ", ["W", "E"], key="lon_dir")
    
    # Convert to decimal degrees
    lat = lat_value if lat_dir == "N" else -lat_value
    lon = lon_value if lon_dir == "E" else -lon_value
    
    st.info(f"📍 Location: **{lat_value}°{lat_dir}, {lon_value}°{lon_dir}** → Decimal: ({lat:.4f}, {lon:.4f})")
    
    if st.button("🔍 Fetch & Analyze Satellite Image", type="primary", use_container_width=True):
        if not model:
            st.error("❌ Model not loaded. Please refresh the page.")
        elif not client_id or not client_secret:
            st.warning("⚠️ Please provide Sentinel Hub credentials")
        else:
            try:
                with st.spinner("🛰️ Fetching Sentinel-2 imagery..."):
                    image = fetch_sentinel2_image(lat, lon, client_id, client_secret)
                
                with st.spinner("🤖 Running AI analysis..."):
                    img_array = preprocess_image(image)
                    label, confidence, stats = predict(img_array)
                
                st.markdown("---")
                display_results(image, label, confidence, stats, show_comparison)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Ensure your Sentinel Hub credentials are correct and you have API access.")

# =======================
# TAB 2 — Upload
# =======================
with tab2:
    st.subheader("Upload Your Own Image")
    st.markdown("Upload aerial or satellite imagery to detect wildfires")
    
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )
    
    if uploaded and model:
        try:
            image = Image.open(uploaded).convert("RGB")
            
            with st.spinner("🤖 Analyzing image..."):
                img_array = preprocess_image(image)
                label, confidence, stats = predict(img_array)
            
            st.markdown("---")
            display_results(image, label, confidence, stats, show_comparison)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    
    elif uploaded and not model:
        st.error("❌ Model not loaded. Please refresh the page.")

# =======================
# TAB 3 — About
# =======================
with tab3:
    st.markdown("""
    ## 🔥 About This System
    
    This wildfire detection system uses deep learning (EfficientNetB4) to analyze satellite 
    and aerial imagery for signs of active wildfires.
    
    ### 🎯 Features
    - **Real-time Satellite Analysis**: Fetch latest Sentinel-2 imagery by coordinates
    - **Custom Image Upload**: Analyze your own aerial/satellite images
    - **AI-Powered Detection**: Deep learning model trained on wildfire datasets
    - **Detailed Metrics**: Get probability scores, confidence levels, and risk assessment
    
    ### 📊 Model Output Parameters
    - **Wildfire Probability**: Likelihood of wildfire presence (0-100%)
    - **Confidence Score**: Model's certainty in its prediction
    - **Risk Level**: LOW, MODERATE, or HIGH based on probability
    - **Decision Threshold**: 0.5 (50% probability cutoff)
    
    ### 🛰️ How to Use Satellite Analysis
    1. Get free Sentinel Hub credentials at [sentinel-hub.com](https://www.sentinel-hub.com/)
    2. Enter coordinates with N/S/E/W directions
    3. Click "Fetch & Analyze" to get latest satellite imagery
    
    ### 📍 Example Coordinates
    - **Los Angeles, USA**: 34.0°N, 118.2°W
    - **Sydney, Australia**: 33.8°S, 151.2°E
    - **Amazon Rainforest**: 3.4°S, 62.2°W
    
    ### ⚠️ Important Disclaimer
    This is an **educational tool** and should **NOT** be used as the sole method for wildfire 
    detection in critical situations. Always verify with official sources and emergency services.
    
    ### 🔧 Technical Details
    - **Model**: EfficientNetB4 (Transfer Learning)
    - **Input Size**: 224×224×3 RGB
    - **Output**: Binary classification (Sigmoid activation)
    - **Data Source**: Sentinel-2 L2A (10m resolution)
    
    ### 📝 Notes
    - Satellite imagery depends on cloud coverage and acquisition schedule
    - Model accuracy may vary based on image quality and fire characteristics
    - Image comparison shows preprocessing steps applied before model inference
    """)

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🛰️ Powered by Sentinel-2 | 🤖 TensorFlow & EfficientNetB4<br>
    Educational use only • Not for emergency decision-making
</div>
""", unsafe_allow_html=True)
