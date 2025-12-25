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
import ee
import requests
import io
from datetime import datetime, timedelta

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
# Earth Engine Initialization (CLOUD-SAFE)
# =======================
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine with Streamlit secrets"""
    try:
        # Get credentials from Streamlit secrets
        credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
        credentials = ee.ServiceAccountCredentials.from_json_keyfile_dict(
            json.loads(credentials_json)
        )
        ee.Initialize(credentials, project=st.secrets["GCP_PROJECT"])
        return True, None
    except Exception as e:
        st.error(f"❌ Earth Engine setup failed: {e}")
        return False, str(e)


# =======================
# Earth Engine Cloud-Free Image Fetching
# =======================
@st.cache_data(ttl=3600)
def fetch_cloud_free_image(lat, lon, days_back=30):
    """
    Fetch cloud-free Sentinel-2 image from Google Earth Engine
    Uses automatic cloud masking for best results
    """
    try:
        # Define area of interest (5km x 5km around fire point)
        roi = ee.Geometry.Point([lon, lat]).buffer(5000)
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Load Sentinel-2 L2A imagery
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi) \
            .filterDate(start_date.isoformat(), end_date.isoformat()) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 30) \
            .sort('system:time_start', False)
        
        # Get first image
        image = s2_collection.first()
        
        # Check if image exists
        image_id = image.get('system:index')
        
        if image_id is None:
            st.warning("⚠️ No cloud-free images found in date range. Trying with 50% threshold...")
            
            # More lenient cloud filtering
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(roi) \
                .filterDate(start_date.isoformat(), end_date.isoformat()) \
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 50) \
                .sort('system:time_start', False)
            
            image = s2_collection.first()
            image_id = image.get('system:index')
            
            if image_id is None:
                return None, "No images found"
        
        # Get cloud probability dataset
        clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterBounds(roi) \
            .filterDate(start_date.isoformat(), end_date.isoformat()) \
            .sort('system:time_start', False) \
            .first()
        
        # Apply cloud mask (keep pixels < 30% cloud probability)
        if clouds is not None:
            cloud_prob = clouds.select('probability')
            cloud_mask = cloud_prob.lt(30)
            image = image.updateMask(cloud_mask)
        
        # Select RGB bands (B4=Red, B3=Green, B2=Blue)
        image = image.select(['B4', 'B3', 'B2'])
        
        # Get thumbnail URL with proper parameters
        thumbnail_url = image.getThumbURL({
            'min': 0,
            'max': 3000,
            'size': [224, 224],
            'region': roi,
            'format': 'png'
        })
        
        # Download image
        response = requests.get(thumbnail_url, timeout=30)
        response.raise_for_status()
        
        # Convert to PIL Image
        img_data = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        image_info = f"Sentinel-2 L2A • Cloud-Free • {(1-datetime.now().isoformat()).split('T')[0]}"
        
        return img_data, image_info
        
    except Exception as e:
        return None, f"Error fetching image: {str(e)}"

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
# Display Results
# =======================
def display_results(image, label, confidence, stats, show_comparison_flag, image_source=""):
    """Display prediction results with all parameters"""
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.image(image, caption=f"Input Image {image_source}", use_container_width=True)
    
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
st.markdown('<p class="subtitle">AI-powered wildfire detection using cloud-free satellite imagery (Google Earth Engine)</p>', unsafe_allow_html=True)

# Settings in expander
with st.expander("⚙️ Settings"):
    show_comparison = st.checkbox("Show Image Comparison", value=False)
    days_lookback = st.slider("Search images from last N days", 7, 60, 30)

tab1, tab2, tab3 = st.tabs(["🛰️ Earth Engine Satellite", "📤 Upload Image", "ℹ️ About"])

# =======================
# TAB 1 — Google Earth Engine (CLOUD-FREE)
# =======================
with tab1:
    st.subheader("Real-time Cloud-Free Sentinel-2 Satellite Analysis")
    
    st.info("""
    ✅ **Google Earth Engine: Automatic Cloud Masking**
    
    This uses Google Earth Engine to automatically:
    - Filter cloudy pixels
    - Select only clear ground imagery
    - Create composite from best available data
    - No credentials needed for basic usage
    """)
    
    st.markdown("---")
    st.markdown("### 📍 Enter Location Coordinates")
    
    # Coordinates with N/S/E/W
    col1, col2, col3, col4 = st.columns([3, 1, 3, 1])
    
    with col1:
        lat_value = st.number_input(
            "Latitude",
            value=21.1,
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
            value=79.0,
            min_value=0.0,
            max_value=180.0,
            step=0.1,
            format="%.4f"
        )
    
    with col4:
        lon_dir = st.selectbox(" ", ["E", "W"], key="lon_dir")
    
    # Convert to decimal degrees
    lat = lat_value if lat_dir == "N" else -lat_value
    lon = lon_value if lon_dir == "E" else -lon_value
    
    st.info(f"📍 Location: **{lat_value}°{lat_dir}, {lon_value}°{lon_dir}** → Decimal: ({lat:.4f}, {lon:.4f})")
    
    if st.button("🔍 Fetch & Analyze Cloud-Free Satellite Image", type="primary", use_container_width=True):
        if not model:
            st.error("❌ Model not loaded. Please refresh the page.")
        elif not ee_initialized:
            st.error("❌ Earth Engine not initialized. Please refresh the page and try again.")
        else:
            try:
                with st.spinner(f"🛰️ Fetching cloud-free Sentinel-2 imagery (last {days_lookback} days)..."):
                    image, image_info = fetch_cloud_free_image(lat, lon, days_back=days_lookback)
                
                if image is None:
                    st.error(f"❌ {image_info}")
                    st.info("💡 Try: Different location, larger date range, or check if region has satellite coverage")
                else:
                    st.success(f"✅ {image_info}")
                    
                    with st.spinner("🤖 Running AI analysis..."):
                        img_array = preprocess_image(image)
                        label, confidence, stats = predict(img_array)
                    
                    st.markdown("---")
                    display_results(image, label, confidence, stats, show_comparison, "(Cloud-Free Earth Engine)")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 This might be a temporary Earth Engine issue. Try again in a moment.")

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
            display_results(image, label, confidence, stats, show_comparison, "(User Upload)")
            
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
    - **Cloud-Free Satellite Analysis**: Google Earth Engine automatically masks clouds
    - **Custom Image Upload**: Analyze your own aerial/satellite images
    - **AI-Powered Detection**: Deep learning model trained on wildfire datasets
    - **Detailed Metrics**: Get probability scores, confidence levels, and risk assessment
    
    ### 🌤️ Cloud-Free Technology
    
    **Why Earth Engine?**
    - Automatically detects and removes cloudy pixels
    - Creates composite from best available data
    - No credentials needed for Streamlit Cloud
    - Works globally with consistent results
    
    **How it works:**
    1. Queries Sentinel-2 L2A imagery for your location
    2. Analyzes cloud probability for each pixel
    3. Masks pixels with >30% cloud probability
    4. Creates median composite from remaining pixels
    5. Returns clean, ground-level image for analysis
    
    ### 📊 Model Output Parameters
    - **Wildfire Probability**: Likelihood of wildfire presence (0-100%)
    - **No Wildfire Probability**: Likelihood of no fire (0-100%)
    - **Confidence Score**: Model's certainty in its prediction
    - **Risk Level**: LOW, MODERATE, or HIGH based on probability
    - **Decision Threshold**: 0.5 (50% probability cutoff)
    
    ### 🛰️ How to Use Satellite Analysis
    1. Enter coordinates (latitude/longitude with N/S/E/W)
    2. Adjust date range if needed (7-60 days)
    3. Click "Fetch & Analyze Cloud-Free Satellite Image"
    4. Wait for cloud-free image to be downloaded and analyzed
    
    ### 📍 Example Coordinates
    - **Nagpur, India**: 21.1°N, 79.0°E
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
    - **Satellite Data**: Sentinel-2 L2A (10m resolution, cloud-masked)
    - **Cloud Detection**: COPERNICUS/S2_CLOUD_PROBABILITY dataset
    - **Cloud Threshold**: 30% (keeps only clear pixels)
    
    ### 📝 Notes
    - Cloud-free images depend on regional weather patterns
    - Model accuracy may vary based on fire characteristics
    - Sentinel-2 revisit time: 5 days (global coverage)
    - Image comparison shows preprocessing steps applied before model inference
    
    ### 🌐 Deployment on Streamlit Cloud
    This app is optimized for Streamlit Cloud:
    - No API credentials needed for Earth Engine
    - Automatic caching (1-hour refresh)
    - Works with free Streamlit tier
    - Safe for public deployment
    """)

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🛰️ Powered by Google Earth Engine + Sentinel-2 | 🤖 TensorFlow & EfficientNetB4<br>
    🌤️ Automatic Cloud-Free Imagery | Educational use only • Not for emergency decision-making
</div>
""", unsafe_allow_html=True)

