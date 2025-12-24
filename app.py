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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        font-size: 16px;
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
        with st.spinner("📥 Downloading model from Google Drive (one-time)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)

    # Sanity check
    if model.output_shape[-1] != 1:
        raise ValueError("Model must be binary sigmoid with output shape (*, 1)")

    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {e}")
    model = None


# =======================
# Image Preprocessing (FIXED - Matches Training)
# =======================
def preprocess_image(image, debug=False):
    """
    Preprocess image to EXACTLY match training preprocessing
    Uses the same preprocess_input function from Keras applications
    """
    if image is None:
        return None

    # Resize with LANCZOS (high quality)
    image_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # Convert to array (matches img_to_array behavior)
    img = img_to_array(image_resized)
    
    if debug:
        st.sidebar.write("### 🔍 Preprocessing Debug")
        st.sidebar.write(f"Original shape: {img.shape}")
        st.sidebar.write(f"Original range: [{img.min():.2f}, {img.max():.2f}]")
    
    # Select preprocessing based on model architecture
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
        index=0,
        help="Select the base model architecture used during training"
    )
    
    # Apply the correct preprocessing based on architecture
    if preprocessing_method == "EfficientNetB4 (Default)":
        # EfficientNet preprocessing: scales to 0-1 range with ImageNet normalization
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "ResNet50/ResNet101/ResNet152":
        # ResNet preprocessing: caffe mode (BGR, mean subtraction, no scaling)
        from tensorflow.keras.applications.resnet50 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "VGG16/VGG19":
        # VGG preprocessing: caffe mode (BGR, mean subtraction)
        from tensorflow.keras.applications.vgg16 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "MobileNetV2":
        # MobileNetV2: range -1 to 1
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "InceptionV3/Xception":
        # Inception/Xception: range -1 to 1
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        img = preprocess_input(img)
    elif preprocessing_method == "DenseNet":
        # DenseNet: caffe mode
        from tensorflow.keras.applications.densenet import preprocess_input
        img = preprocess_input(img)
    
    if debug:
        st.sidebar.write(f"Processed range: [{img.min():.2f}, {img.max():.2f}]")
        st.sidebar.write(f"Mean: {img.mean():.4f}")
        st.sidebar.write(f"Std: {img.std():.4f}")
    
    if img.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Invalid image shape: {img.shape}")

    return np.expand_dims(img, axis=0)

# =======================
# Prediction with Debug
# =======================
def predict(image_array, debug=False):
    """
    Make prediction with detailed output
    """
    if debug:
        st.sidebar.write("### Model Input Debug")
        st.sidebar.write(f"Input shape: {image_array.shape}")
        st.sidebar.write(f"Input range: [{image_array.min():.4f}, {image_array.max():.4f}]")
        st.sidebar.write(f"Input mean: {image_array.mean():.4f}")
        st.sidebar.write(f"Input std: {image_array.std():.4f}")
    
    preds = model.predict(image_array, verbose=0)

    if preds.ndim != 2 or preds.shape[1] != 1:
        raise ValueError("Model output must be (batch, 1) sigmoid")

    prob = float(preds[0][0])
    
    if debug:
        st.sidebar.write(f"Raw model output: {prob:.6f}")
    
    label = int(prob >= 0.5)
    confidence = prob if label == 1 else 1 - prob

    return label, confidence, prob

# =======================
# Sentinel-2 Fetch
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
# Modern Card Visualization
# =======================
# =======================
# Modern Card Visualization
# =======================
def create_confidence_chart(raw_prob):
    """Create a sleek, modern visualization with smooth gradients"""
    fig = plt.figure(figsize=(12, 7), facecolor='#f8f9fa')
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], height_ratios=[2, 1, 1],
                          hspace=0.4, wspace=0.3)
    
    # Main circular gauge
    ax_gauge = fig.add_subplot(gs[0, :])
    ax_gauge.axis('off')
    
    # Create smooth circular progress bar
    theta = np.linspace(0, np.pi, 100)
    r_outer = 1
    r_inner = 0.7
    
    # Background arc (gray)
    x_bg_outer = r_outer * np.cos(theta)
    y_bg_outer = r_outer * np.sin(theta)
    x_bg_inner = r_inner * np.cos(theta)
    y_bg_inner = r_inner * np.sin(theta)
    
    ax_gauge.fill_between(x_bg_outer, y_bg_outer, 0, alpha=0.1, color='gray')
    
    # Colored progress arc
    theta_filled = np.linspace(0, np.pi * raw_prob, 100)
    x_filled_outer = r_outer * np.cos(theta_filled)
    y_filled_outer = r_outer * np.sin(theta_filled)
    x_filled_inner = r_inner * np.cos(theta_filled)
    y_filled_inner = r_inner * np.sin(theta_filled)
    
    # Gradient colors based on probability
    if raw_prob < 0.3:
        colors = ['#2ecc71', '#27ae60']
    elif raw_prob < 0.7:
        colors = ['#f39c12', '#e67e22']
    else:
        colors = ['#e74c3c', '#c0392b']
    
    # Create gradient effect
    for i in range(len(theta_filled)-1):
        alpha = 0.3 + 0.7 * (i / len(theta_filled))
        color_idx = int((i / len(theta_filled)) * (len(colors) - 1))
        ax_gauge.fill_between(
            [x_filled_outer[i], x_filled_outer[i+1]], 
            [y_filled_outer[i], y_filled_outer[i+1]], 
            [x_filled_inner[i], x_filled_inner[i+1]],
            color=colors[min(color_idx, len(colors)-1)],
            alpha=alpha
        )
    
    # Center text
    ax_gauge.text(0, 0.3, f'{raw_prob*100:.1f}%', 
                ha='center', va='center', fontsize=56, fontweight='bold',
                color='#2c3e50')
    ax_gauge.text(0, -0.1, 'WILDFIRE PROBABILITY', 
                ha='center', va='center', fontsize=13, fontweight='600',
                color='#7f8c8d', letterSpacing=2)
    
    # Pointer/needle
    angle = np.pi * (1 - raw_prob)
    needle_length = 0.85
    ax_gauge.plot([0, needle_length * np.cos(angle)], 
                 [0, needle_length * np.sin(angle)],
                 color='#2c3e50', linewidth=4, solid_capstyle='round')
    ax_gauge.plot(0, 0, 'o', color='#2c3e50', markersize=15)
    ax_gauge.plot(0, 0, 'o', color='white', markersize=8)
    
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-0.3, 1.3)
    ax_gauge.set_aspect('equal')
    
    # Risk Level Card
    ax_risk = fig.add_subplot(gs[1, :2])
    ax_risk.axis('off')
    
    if raw_prob < 0.3:
        risk_level = "LOW RISK"
        risk_color = "#2ecc71"
        risk_emoji = "✓"
    elif raw_prob < 0.7:
        risk_level = "MODERATE RISK"
        risk_color = "#f39c12"
        risk_emoji = "⚠"
    else:
        risk_level = "HIGH RISK"
        risk_color = "#e74c3c"
        risk_emoji = "⚠"
    
    # Rounded rectangle
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                          boxstyle="round,pad=0.05",
                          facecolor=risk_color, alpha=0.15,
                          edgecolor=risk_color, linewidth=3)
    ax_risk.add_patch(rect)
    
    ax_risk.text(0.5, 0.6, risk_emoji, ha='center', va='center',
                fontsize=40, color=risk_color)
    ax_risk.text(0.5, 0.3, risk_level, ha='center', va='center',
                fontsize=20, fontweight='bold', color=risk_color)
    ax_risk.set_xlim(0, 1)
    ax_risk.set_ylim(0, 1)
    
    # Confidence Card
    ax_conf = fig.add_subplot(gs[1, 2])
    ax_conf.axis('off')
    
    confidence = max(raw_prob, 1 - raw_prob)
    conf_color = "#3498db"
    
    rect_conf = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=conf_color, alpha=0.15,
                               edgecolor=conf_color, linewidth=3)
    ax_conf.add_patch(rect_conf)
    
    ax_conf.text(0.5, 0.6, f'{confidence*100:.0f}%', ha='center', va='center',
                fontsize=28, fontweight='bold', color=conf_color)
    ax_conf.text(0.5, 0.25, 'Confidence', ha='center', va='center',
                fontsize=11, color='#7f8c8d', fontweight='600')
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(0, 1)
    
    # Horizontal probability bar
    ax_bar = fig.add_subplot(gs[2, :])
    ax_bar.axis('off')
    
    # Create smooth gradient bar
    bar_gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_bar.imshow(bar_gradient, extent=[0, 1, 0, 1], aspect='auto',
                  cmap='RdYlGn_r', alpha=0.4)
    
    # Current position marker
    ax_bar.axvline(raw_prob, color='#2c3e50', linewidth=3, alpha=0.8)
    ax_bar.plot(raw_prob, 0.5, 'o', color='#2c3e50', markersize=14, 
               markeredgecolor='white', markeredgewidth=2)
    
    # Labels
    ax_bar.text(0, -0.3, 'Safe', ha='left', va='top', fontsize=10, 
               color='#27ae60', fontweight='600')
    ax_bar.text(1, -0.3, 'Danger', ha='right', va='top', fontsize=10,
               color='#c0392b', fontweight='600')
    ax_bar.text(0.5, -0.3, 'Threshold', ha='center', va='top', fontsize=10,
               color='#7f8c8d', fontweight='600')
    
    ax_bar.set_xlim(-0.05, 1.05)
    ax_bar.set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    return fig
def show_image_comparison(original, processed):
    """Show original vs processed image"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(original)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Show processed (remove batch dimension and denormalize for display)
    processed_display = processed[0]
    # Normalize to 0-1 for display
    processed_display = (processed_display - processed_display.min()) / (processed_display.max() - processed_display.min())
    ax2.imshow(processed_display)
    ax2.set_title("Processed Image (Model Input)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# =======================
# Sidebar Settings
# =======================
st.sidebar.title("⚙️ Settings")

debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, 
                                  help="Show detailed preprocessing and prediction info")
show_comparison = st.sidebar.checkbox("🖼️ Show Image Comparison", value=False,
                                      help="Display original vs processed image")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
if model:
    st.sidebar.info(f"Input Shape: {model.input_shape}\nOutput Shape: {model.output_shape}")

# =======================
# Main UI
# =======================
st.markdown('<p class="main-header">🔥 Wildfire Detection System</p>', unsafe_allow_html=True)
st.markdown("### Advanced AI-powered wildfire detection using satellite imagery and deep learning")

tab1, tab2, tab3 = st.tabs(["🛰️ Sentinel-2 Analysis", "📤 Upload Image", "ℹ️ About"])

# =======================
# TAB 1 — Sentinel-2
# =======================
with tab1:
    st.subheader("Real-time Satellite Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        client_id = st.text_input("Sentinel Hub Client ID", type="password",
                                   help="Get your credentials at https://www.sentinel-hub.com/")
        lat = st.number_input("Latitude", value=21.1, step=0.1, format="%.4f")
        
    with col2:
        client_secret = st.text_input("Sentinel Hub Client Secret", type="password")
        lon = st.number_input("Longitude", value=79.0, step=0.1, format="%.4f")

    if st.button("🔍 Fetch & Analyze", type="primary", use_container_width=True):
        if not model:
            st.error("❌ Model not loaded. Please check the sidebar for errors.")
        elif not client_id or not client_secret:
            st.warning("⚠️ Please provide Sentinel Hub credentials")
        else:
            try:
                with st.spinner("🛰️ Fetching Sentinel-2 image..."):
                    image = fetch_sentinel2_image(lat, lon, client_id, client_secret)

                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image, caption="Sentinel-2 True Color Image", use_container_width=True)

                with st.spinner("🤖 Analyzing image..."):
                    img_array = preprocess_image(image, debug=debug_mode)
                    label, confidence, raw_prob = predict(img_array, debug=debug_mode)

                with col2:
                    if label == 1:
                        st.error(f"### 🔥 Wildfire Detected!\n**Confidence: {confidence*100:.2f}%**")
                        st.warning("⚠️ Immediate action may be required")
                    else:
                        st.success(f"### ✅ No Wildfire Detected\n**Confidence: {confidence*100:.2f}%**")
                        st.info("🌲 Area appears safe")
                
                # Confidence chart
                fig = create_confidence_chart(raw_prob)
                st.pyplot(fig)
                plt.close()
                
                # Image comparison
                if show_comparison:
                    st.markdown("---")
                    st.subheader("Image Processing Comparison")
                    fig_comp = show_image_comparison(image, img_array)
                    st.pyplot(fig_comp)
                    plt.close()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# =======================
# TAB 2 — Upload
# =======================
with tab2:
    st.subheader("Upload Your Own Image")
    
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload an aerial or satellite image to detect wildfires"
    )

    if uploaded and model:
        image = Image.open(uploaded).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🤖 Analyzing image..."):
            img_array = preprocess_image(image, debug=debug_mode)
            label, confidence, raw_prob = predict(img_array, debug=debug_mode)

        with col2:
            if label == 1:
                st.error(f"### 🔥 Wildfire Detected!\n**Confidence: {confidence*100:.2f}%**")
                st.warning("⚠️ Immediate action may be required")
            else:
                st.success(f"### ✅ No Wildfire Detected\n**Confidence: {confidence*100:.2f}%**")
                st.info("🌲 Area appears safe")
        
        # Confidence chart
        fig = create_confidence_chart(raw_prob)
        st.pyplot(fig)
        plt.close()
        
        # Image comparison
        if show_comparison:
            st.markdown("---")
            st.subheader("Image Processing Comparison")
            fig_comp = show_image_comparison(image, img_array)
            st.pyplot(fig_comp)
            plt.close()
            
    elif uploaded and not model:
        st.error("❌ Model not loaded. Cannot make predictions.")

# =======================
# TAB 3 — About
# =======================
with tab3:
    st.markdown("""
    ## About This System
    
    This wildfire detection system uses deep learning to analyze satellite and aerial imagery 
    for signs of active wildfires. The system is trained on a large dataset of wildfire images 
    and can detect fire patterns with high accuracy.
    
    ### 🎯 Features
    - Real-time Sentinel-2 satellite image analysis
    - Upload custom images for detection
    - Multiple preprocessing options for different image types
    - Confidence scoring and visualization
    - Debug mode for troubleshooting
    
    ### 🔧 How to Use
    1. **Sentinel-2 Tab**: Enter coordinates and your Sentinel Hub API credentials
    2. **Upload Tab**: Upload any aerial/satellite image
    3. **Settings**: Toggle debug mode and preprocessing methods in the sidebar
    
    ### ⚙️ Troubleshooting
    If you're getting the same prediction for different images:
    - Try different preprocessing methods in the sidebar
    - Enable debug mode to see input statistics
    - Check that images are different (use image comparison)
    - Ensure model was trained with the same preprocessing
    
    ### 📊 Model Information
    - Input Size: 224x224x3
    - Output: Binary classification (Wildfire/No Wildfire)
    - Threshold: 0.5
    
    ### ⚠️ Disclaimer
    This is an educational tool and should not be used as the sole method for wildfire detection 
    in critical situations. Always verify with official sources and emergency services.
    """)

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🛰️ Powered by Sentinel-2 L2A | 🤖 TensorFlow Deep Learning</p>
    <p style='font-size: 0.9em;'>Cloud presence depends on acquisition date • Educational use only</p>
</div>
""", unsafe_allow_html=True)

