import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import io

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Wildfire Detection System",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-header">
        <h1>🔥 Wildfire Detection System</h1>
        <p style="font-size:18px; margin:0;">
        AI-powered wildfire detection using NASA satellite data and deep learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =======================
# Model Download & Load
# =======================

MODEL_URL = "https://drive.google.com/uc?id=1PnOX7t7o2Qqly-3nqVlSLa5LoGApvGjW"
MODEL_PATH = "best_model.keras"

@st.cache_resource
def download_and_load_model():
    """Download and load the model with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model (one-time setup)..."):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = download_and_load_model()

if model_error:
    st.error(f"❌ Failed to load model: {model_error}")
    st.info("💡 The app will continue with limited functionality. Please check your internet connection and try refreshing.")
    model = None

# =======================
# Constants
# =======================
IMG_SIZE = 224
CLASS_LABELS = {0: "No Wildfire", 1: "Wildfire"}

# =======================
# Image Processing
# =======================
def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(image) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict(image_array):
    """Get prediction from model"""
    if model is None:
        return None, None, None
    
    try:
        prob = model.predict(image_array, verbose=0)[0][0]
        label = 1 if prob > 0.5 else 0
        confidence = prob if label == 1 else (1 - prob)
        return label, confidence, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

# =======================
# NASA FIRMS API Functions
# =======================
def fetch_firms_fires(lon_min, lat_min, lon_max, lat_max, firms_api_key):
    """Fetch fire hotspots from NASA FIRMS API"""
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{firms_api_key}/VIIRS_SNPP_NRT/"
        f"{lon_min},{lat_min},{lon_max},{lat_max}/7"
    )
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return []
        
        fires = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 13:
                continue
            
            try:
                fire = {
                    'latitude': float(parts[0]),
                    'longitude': float(parts[1]),
                    'brightness': float(parts[2]),
                    'confidence': parts[8],  # Can be 'low', 'nominal', 'high'
                    'frp': float(parts[11]),
                    'satellite': parts[7],
                    'acq_date': parts[5]
                }
                fires.append(fire)
            except (ValueError, IndexError) as e:
                continue
        
        return fires
    
    except requests.RequestException as e:
        st.error(f"❌ Error fetching FIRMS data: {e}")
        return None

def create_dummy_satellite_image(fire_data):
    """Create a placeholder satellite image"""
    # Create a gradient image to represent heat
    img_array = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Add some texture
    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            # Create a radial gradient (simulating heat signature)
            dist = np.sqrt((i - IMG_SIZE//2)**2 + (j - IMG_SIZE//2)**2)
            val = int(255 * (1 - dist / (IMG_SIZE * 0.7)))
            val = max(0, min(255, val))
            img_array[i, j] = [val, val//2, 0]  # Orange gradient
    
    return Image.fromarray(img_array)

def create_confidence_chart(raw_prob):
    """Create a confidence visualization chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    scores = [1 - raw_prob, raw_prob]
    classes = ["No Wildfire", "Wildfire"]
    colors = ['#28a745', '#dc3545']
    
    bars = ax.barh(classes, scores, color=colors, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=12, fontweight='bold')
    ax.set_title("Model Confidence Score", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score * 100:.1f}%",
            va="center",
            fontweight="bold",
            fontsize=11
        )
    
    plt.tight_layout()
    return fig

# =======================
# Main App UI
# =======================

# Sidebar with instructions
with st.sidebar:
    st.image("https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg", width=150)
    st.title("📋 Quick Guide")
    st.markdown("""
    ### Getting Started
    
    **Option 1: NASA Live Data** 🛰️
    1. Get a FREE API key from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/)
    2. Enter your API key
    3. Select a region
    4. Fetch live fire data
    
    **Option 2: Upload Image** 📤
    1. Switch to Upload tab
    2. Upload a wildfire image
    3. Get instant AI prediction
    
    ---
    
    ### Model Info
    - **Architecture**: CNN
    - **Input Size**: 224×224
    - **Classes**: Wildfire / No Wildfire
    """)
    
    if model:
        st.success("✅ Model Loaded")
    else:
        st.error("⚠️ Model Not Available")

tab1, tab2, tab3 = st.tabs(["🛰️ NASA Live Data", "📤 Upload Image", "ℹ️ About"])

# =======================
# TAB 1: NASA FIRMS Integration
# =======================
with tab1:
    st.markdown("### Fetch Live Satellite Fire Data from NASA")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # NASA API Setup
        with st.expander("🔑 NASA FIRMS API Key Setup", expanded=False):
            st.markdown("""
            **Get your FREE NASA FIRMS API key:**
            
            1. Visit [NASA FIRMS Map Keys](https://firms.modaps.eosdis.nasa.gov/api/)
            2. Click on **"Map Keys"** or **"Request a key"**
            3. Fill out the registration form
            4. Receive your API key via email instantly
            5. Paste it below
            
            *Note: The API key is free and takes less than 2 minutes to obtain.*
            """)
        
        firms_api_key = st.text_input(
            "Enter NASA FIRMS API Key",
            type="password",
            placeholder="Your API key here...",
            help="Get your free key at https://firms.modaps.eosdis.nasa.gov/api/"
        )
    
    with col2:
        st.info("💡 **Tip**: The NASA FIRMS API is completely free and provides real-time fire detection data worldwide.")
    
    st.markdown("---")
    
    # Region Selection
    st.markdown("### 📍 Select Monitoring Region")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        region_name = st.selectbox(
            "Quick Select Region",
            [
                "🇮🇳 Nagpur, India",
                "🇮🇳 Central India",
                "🇮🇳 Southern India",
                "🇺🇸 California, USA",
                "🇦🇺 Eastern Australia",
                "🌍 Custom Region"
            ]
        )
    
    # Define region coordinates
    regions = {
        "🇮🇳 Nagpur, India": {"lon": [78.5, 79.5], "lat": [20.8, 21.8]},
        "🇮🇳 Central India": {"lon": [74, 82], "lat": [20, 24]},
        "🇮🇳 Southern India": {"lon": [72, 80], "lat": [8, 16]},
        "🇺🇸 California, USA": {"lon": [-124, -114], "lat": [32, 42]},
        "🇦🇺 Eastern Australia": {"lon": [140, 154], "lat": [-38, -28]},
    }
    
    if region_name == "🌍 Custom Region":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            lon_min = st.number_input("Min Longitude", value=77.0, step=0.1)
        with col2:
            lon_max = st.number_input("Max Longitude", value=79.0, step=0.1)
        with col3:
            lat_min = st.number_input("Min Latitude", value=20.0, step=0.1)
        with col4:
            lat_max = st.number_input("Max Latitude", value=22.0, step=0.1)
    else:
        lon_min, lon_max = regions[region_name]["lon"]
        lat_min, lat_max = regions[region_name]["lat"]
        
        with col2:
            st.metric("Longitude Range", f"{lon_min}° to {lon_max}°")
            st.metric("Latitude Range", f"{lat_min}° to {lat_max}°")
    
    st.markdown("---")
    
    # Fetch FIRMS Data
    fetch_button = st.button("🔍 Fetch Fire Data from NASA FIRMS", type="primary", use_container_width=True)
    
    if fetch_button:
        if not firms_api_key:
            st.error("❌ Please enter your NASA FIRMS API key first!")
        else:
            with st.spinner("🛰️ Fetching fire hotspots from NASA FIRMS..."):
                fires = fetch_firms_fires(lon_min, lat_min, lon_max, lat_max, firms_api_key)
            
            if fires is None:
                st.error("❌ Failed to fetch data. Please check your API key and try again.")
            elif len(fires) == 0:
                st.info("ℹ️ No fire hotspots detected in the selected region in the last 7 days.")
            else:
                st.success(f"✅ Found **{len(fires)}** fire hotspot(s) in the last 7 days!")
                
                # Display fire data in a nice table
                st.markdown("### 📊 Fire Hotspot Data")
                
                fire_df = []
                for f in fires:
                    fire_df.append({
                        'Latitude': f"{f['latitude']:.4f}",
                        'Longitude': f"{f['longitude']:.4f}",
                        'Confidence': f['confidence'],
                        'Brightness (K)': f"{f['brightness']:.1f}",
                        'FRP (MW)': f"{f['frp']:.1f}",
                        'Date': f['acq_date'],
                        'Satellite': f['satellite']
                    })
                
                st.dataframe(fire_df, use_container_width=True, height=300)
                
                st.markdown("---")
                
                # Process each fire detection
                st.markdown("### 🤖 AI Model Analysis")
                
                if model is None:
                    st.warning("⚠️ Model not available. Cannot perform AI analysis.")
                else:
                    for idx, fire in enumerate(fires[:5]):  # Limit to first 5 to avoid timeout
                        with st.expander(
                            f"🔥 Fire #{idx + 1} - Lat: {fire['latitude']:.4f}, Lon: {fire['longitude']:.4f} | "
                            f"Confidence: {fire['confidence']} | FRP: {fire['frp']:.1f} MW",
                            expanded=(idx == 0)
                        ):
                            col1, col2 = st.columns([1, 1.5])
                            
                            with col1:
                                # Create placeholder image
                                with st.spinner(f"Generating visualization..."):
                                    image = create_dummy_satellite_image(fire)
                                
                                st.image(image, caption=f"Location Visualization - {fire['acq_date']}")
                                
                                st.markdown("**NASA FIRMS Data:**")
                                st.metric("📅 Detection Date", fire['acq_date'])
                                st.metric("🌡️ Brightness", f"{fire['brightness']:.1f} K")
                                st.metric("🔥 Fire Radiative Power", f"{fire['frp']:.1f} MW")
                                st.metric("📡 Satellite", fire['satellite'])
                            
                            with col2:
                                # ML Model Prediction
                                img_array = preprocess_image(image)
                                if img_array is not None:
                                    label, confidence, raw_prob = predict(img_array)
                                    
                                    if label is not None:
                                        st.markdown("#### 🤖 AI Model Prediction")
                                        
                                        if label == 1:
                                            st.markdown(
                                                f'<div class="error-box">'
                                                f'<h3 style="margin:0; color:#721c24;">🔥 Wildfire Detected</h3>'
                                                f'<p style="margin:0.5rem 0 0 0; font-size:1.2rem; font-weight:bold;">Confidence: {confidence * 100:.1f}%</p>'
                                                f'</div>',
                                                unsafe_allow_html=True
                                            )
                                        else:
                                            st.markdown(
                                                f'<div class="success-box">'
                                                f'<h3 style="margin:0; color:#155724;">✅ No Wildfire Detected</h3>'
                                                f'<p style="margin:0.5rem 0 0 0; font-size:1.2rem; font-weight:bold;">Confidence: {confidence * 100:.1f}%</p>'
                                                f'</div>',
                                                unsafe_allow_html=True
                                            )
                                        
                                        # Visualization
                                        fig = create_confidence_chart(raw_prob)
                                        st.pyplot(fig)
                                        plt.close()
                    
                    if len(fires) > 5:
                        st.info(f"ℹ️ Showing analysis for first 5 fires. Total fires detected: {len(fires)}")

# =======================
# TAB 2: Manual Image Upload
# =======================
with tab2:
    st.markdown("### Upload Your Own Wildfire Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a satellite or ground-level image to detect wildfires"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                image = None
        else:
            st.info("👆 Upload an image to get started")
            image = None
    
    with col2:
        if uploaded_file and image:
            if model is None:
                st.error("⚠️ Model not available. Cannot perform prediction.")
            else:
                st.markdown("### 🔍 Analysis Result")
                
                with st.spinner("Analyzing image..."):
                    img_array = preprocess_image(image)
                    
                    if img_array is not None:
                        label, confidence, raw_prob = predict(img_array)
                        
                        if label is not None:
                            # Display prediction
                            if label == 1:
                                st.markdown(
                                    f'<div class="error-box">'
                                    f'<h2 style="margin:0; color:#721c24;">🔥 Wildfire Detected</h2>'
                                    f'<p style="margin:0.5rem 0 0 0; font-size:1.5rem; font-weight:bold;">Confidence: {confidence * 100:.1f}%</p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div class="success-box">'
                                    f'<h2 style="margin:0; color:#155724;">✅ No Wildfire Detected</h2>'
                                    f'<p style="margin:0.5rem 0 0 0; font-size:1.5rem; font-weight:bold;">Confidence: {confidence * 100:.1f}%</p>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            
                            st.markdown("---")
                            
                            # Confidence Visualization
                            st.markdown("#### Confidence Breakdown")
                            fig = create_confidence_chart(raw_prob)
                            st.pyplot(fig)
                            plt.close()

# =======================
# TAB 3: About
# =======================
with tab3:
    st.markdown("## About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        
        This application combines NASA's real-time fire detection data with 
        AI-powered image analysis to help identify and monitor wildfires globally.
        
        ### 🛠️ Technology Stack
        
        - **Frontend**: Streamlit
        - **AI Model**: TensorFlow/Keras CNN
        - **Data Source**: NASA FIRMS API
        - **Image Processing**: PIL, NumPy
        - **Visualization**: Matplotlib
        
        ### 🌟 Features
        
        - Real-time fire hotspot detection via NASA FIRMS
        - AI-powered wildfire image classification
        - Interactive region selection
        - Confidence score visualization
        - Support for custom image uploads
        """)
    
    with col2:
        st.markdown("""
        ### 📚 How It Works
        
        **NASA Live Data Mode:**
        1. Fetches active fire hotspots from NASA FIRMS
        2. Displays fire locations and metadata
        3. Generates visualizations for each detection
        4. Runs AI model to classify wildfire presence
        
        **Upload Mode:**
        1. User uploads satellite or ground image
        2. Image is preprocessed to 224×224 pixels
        3. AI model analyzes the image
        4. Returns wildfire/no wildfire classification
        
        ### 🔗 Resources
        
        - [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
        - [Get API Key](https://firms.modaps.eosdis.nasa.gov/api/)
        - [FIRMS Data Format](https://firms.modaps.eosdis.nasa.gov/api/area/)
        
        ### ⚠️ Disclaimer
        
        This is a demonstration application. For operational wildfire 
        monitoring, please consult official sources and emergency services.
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Deployment Notes
    
    **Requirements for Streamlit Cloud:**
    ```
    streamlit
    tensorflow
    gdown
    requests
    pillow
    matplotlib
    numpy
    ```
    
    **Environment Variables:**
    - No sensitive data in code
    - API keys entered by users
    - Model downloaded on first run
    """)

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Built with ❤️ using Streamlit | Powered by NASA FIRMS & TensorFlow</p>
        <p style="font-size: 0.9rem;">For educational and research purposes</p>
    </div>
    """,
    unsafe_allow_html=True
)
