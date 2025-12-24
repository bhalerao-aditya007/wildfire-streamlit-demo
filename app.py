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
        AI-powered wildfire detection using Sentinel-2 satellite imagery and deep learning
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
# Copernicus Data Space Authentication
# =======================
@st.cache_resource
def get_copernicus_token(client_id, client_secret):
    """Get access token from Copernicus Data Space"""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    
    try:
        response = requests.post(token_url, data=data, timeout=10)
        response.raise_for_status()
        token = response.json()["access_token"]
        return token, None
    except requests.RequestException as e:
        return None, str(e)

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
# Copernicus Sentinel-2 API Functions
# =======================
def fetch_sentinel2_image(lat, lon, client_id, client_secret, days_back=7):
    """Fetch Sentinel-2 image from Copernicus Data Space for a specific location"""
    
    # Get token
    token, token_error = get_copernicus_token(client_id, client_secret)
    if token_error:
        st.error(f"❌ Authentication failed: {token_error}")
        return None
    
    try:
        # Define bounding box around the fire location (small area)
        # ~5km x 5km box
        bbox_width = 0.05  # ~5km in degrees
        bbox_height = 0.05
        
        lon_min = lon - bbox_width/2
        lon_max = lon + bbox_width/2
        lat_min = lat - bbox_height/2
        lat_max = lat + bbox_height/2
        
        # Search for Sentinel-2 L2A images
        search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        
        # Date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # OData filter for Sentinel-2 L2A
        filter_str = (
            f"(Name startswith 'S2') and "
            f"(Attributes/any(a:a/Name eq 'productType' and a/OData.COLLECTION.Cast.Value eq 'S2MSI2A')) and "
            f"(ContentDate/Start ge {start_date.isoformat()}T00:00:00.000Z) and "
            f"(ContentDate/Start le {end_date.isoformat()}T23:59:59.999Z) and "
            f"((Footprint geom_intersects POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
            f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))))"
        )
        
        params = {
            "$filter": filter_str,
            "$top": 5,
            "$orderby": "ContentDate/Start desc"
        }
        
        response = requests.get(search_url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        products = response.json().get("value", [])
        
        if not products:
            st.warning("⚠️ No Sentinel-2 images found for this location. Using placeholder image.")
            return create_dummy_satellite_image({"latitude": lat, "longitude": lon})
        
        # Get the most recent product
        product_id = products[0]["Id"]
        product_name = products[0]["Name"]
        
        st.info(f"📡 Using Sentinel-2 image: {product_name[:50]}...")
        
        # Download product (returns ZIP file)
        download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        
        with st.spinner("📥 Downloading Sentinel-2 image..."):
            response = requests.get(download_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save ZIP temporarily
            zip_path = "/tmp/sentinel2_product.zip"
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract and process
        import zipfile
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find the true color image (TCI) or RGB bands
            file_list = zip_ref.namelist()
            
            # Look for TCI_10m.jp2 (True Color Image 10m resolution)
            tci_file = None
            for f in file_list:
                if 'TCI_10m.jp2' in f or 'TCI.jp2' in f:
                    tci_file = f
                    break
            
            if tci_file:
                with zip_ref.open(tci_file) as img_file:
                    import io
                    img_data = io.BytesIO(img_file.read())
                    image = Image.open(img_data).convert("RGB")
                    return image
            else:
                st.warning("⚠️ Could not find image file. Using placeholder.")
                return create_dummy_satellite_image({"latitude": lat, "longitude": lon})
    
    except requests.RequestException as e:
        st.error(f"❌ Error fetching Sentinel-2 data: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
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
    st.image("https://www.copernicus.eu/sites/default/files/2019-09/Copernicus_Logo_RGB_Web.png", width=150)
    st.title("📋 Quick Guide")
    st.markdown("""
    ### Getting Started
    
    **Option 1: Copernicus Sentinel-2** 🛰️
    1. Get FREE credentials from [Copernicus Data Space](https://dataspace.copernicus.eu/)
    2. Enter your Client ID & Secret
    3. Select a region
    4. Fetch Sentinel-2 satellite images
    
    **Option 2: Upload Image** 📤
    1. Switch to Upload tab
    2. Upload a wildfire image
    3. Get instant AI prediction
    
    ---
    
    ### Model Info
    - **Architecture**: CNN
    - **Input Size**: 224×224
    - **Classes**: Wildfire / No Wildfire
    
    ### Copernicus Benefits
    - ✅ 10m resolution (high quality)
    - ✅ Global coverage
    - ✅ Cloud-free imagery
    - ✅ Multiple bands available
    """)
    
    if model:
        st.success("✅ Model Loaded")
    else:
        st.error("⚠️ Model Not Available")

tab1, tab2, tab3 = st.tabs(["🛰️ Sentinel-2 Data", "📤 Upload Image", "ℹ️ About"])

# =======================
# TAB 1: Copernicus Sentinel-2 Integration
# =======================
with tab1:
    st.markdown("### Fetch High-Resolution Satellite Images from Copernicus")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Copernicus API Setup
        with st.expander("🔑 Copernicus Data Space Setup", expanded=False):
            st.markdown("""
            **Get your FREE Copernicus credentials:**
            
            1. Visit [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
            2. Click **"Sign Up"** and create an account
            3. Go to **"User Settings"** → **"OAuth2 Clients"** or **"API Tokens"**
            4. Click **"Create OAuth2 Client"**
            5. Fill form:
               - Name: "Wildfire Detection App"
               - Grant Type: "Client Credentials"
               - Scopes: Select all
            6. Copy your **Client ID** and **Client Secret**
            7. Paste them below
            
            *Note: Completely free! Sentinel-2 images are 10m resolution, perfect for wildfire detection.*
            """)
        
        col_id, col_secret = st.columns(2)
        
        with col_id:
            client_id = st.text_input(
                "Copernicus Client ID",
                type="password",
                placeholder="Your Client ID...",
                help="Get from Copernicus Data Space OAuth2 settings"
            )
        
        with col_secret:
            client_secret = st.text_input(
                "Copernicus Client Secret",
                type="password",
                placeholder="Your Client Secret...",
                help="Get from Copernicus Data Space OAuth2 settings"
            )
    
    with col2:
        st.info("💡 **Why Copernicus?**\n\n✅ 10m resolution\n✅ Free forever\n✅ Global coverage\n✅ 5-day revisit")
    
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
        "🇮🇳 Nagpur, India": {"lon": 79.0, "lat": 21.1},
        "🇮🇳 Central India": {"lon": 78.0, "lat": 22.0},
        "🇮🇳 Southern India": {"lon": 76.0, "lat": 12.0},
        "🇺🇸 California, USA": {"lon": -119.0, "lat": 37.0},
        "🇦🇺 Eastern Australia": {"lon": 147.0, "lat": -33.0},
    }
    
    if region_name == "🌍 Custom Region":
        col1, col2 = st.columns(2)
        with col1:
            lon = st.number_input("Longitude", value=79.0, step=0.1)
        with col2:
            lat = st.number_input("Latitude", value=21.1, step=0.1)
    else:
        lon = regions[region_name]["lon"]
        lat = regions[region_name]["lat"]
        
        with col2:
            st.metric("Longitude", f"{lon:.2f}°")
            st.metric("Latitude", f"{lat:.2f}°")
    
    st.markdown("---")
    
    # Fetch Sentinel-2 Data
    fetch_button = st.button("🔍 Fetch Sentinel-2 Image from Copernicus", type="primary", use_container_width=True)
    
    if fetch_button:
        if not client_id or not client_secret:
            st.error("❌ Please enter your Copernicus Client ID and Secret!")
            st.info("Get free credentials at: https://dataspace.copernicus.eu/")
        else:
            with st.spinner("🛰️ Fetching Sentinel-2 image from Copernicus..."):
                image = fetch_sentinel2_image(lat, lon, client_id, client_secret)
            
            if image:
                st.success(f"✅ Successfully fetched Sentinel-2 image for Lat: {lat:.4f}, Lon: {lon:.4f}")
                
                st.markdown("### 🖼️ Satellite Image Preview")
                st.image(image, caption=f"Sentinel-2 Image - Location: {region_name}", use_column_width=True)
                
                st.markdown("---")
                
                # Process with ML Model
                st.markdown("### 🤖 AI Model Analysis")
                
                if model is None:
                    st.warning("⚠️ Model not available. Cannot perform AI analysis.")
                else:
                    with st.spinner("Analyzing image with AI model..."):
                        img_array = preprocess_image(image)
                        
                        if img_array is not None:
                            label, confidence, raw_prob = predict(img_array)
                            
                            if label is not None:
                                col1, col2 = st.columns([1, 1.5])
                                
                                with col1:
                                    st.markdown("**Location Details:**")
                                    st.metric("📍 Latitude", f"{lat:.4f}°")
                                    st.metric("📍 Longitude", f"{lon:.4f}°")
                                    st.metric("🛰️ Source", "Copernicus Sentinel-2")
                                    st.metric("📐 Resolution", "10m per pixel")
                                
                                with col2:
                                    st.markdown("#### 🔥 Wildfire Detection Result")
                                    
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
                                
                                st.markdown("---")
                                
                                # Confidence Chart
                                st.markdown("#### Confidence Breakdown")
                                fig = create_confidence_chart(raw_prob)
                                st.pyplot(fig)
                                plt.close()

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
        
        This application uses Copernicus Sentinel-2 satellite imagery 
        with AI-powered analysis to help identify and monitor wildfires globally.
        
        ### 🛠️ Technology Stack
        
        - **Frontend**: Streamlit
        - **AI Model**: TensorFlow/Keras CNN
        - **Data Source**: Copernicus Sentinel-2
        - **Image Processing**: PIL, NumPy
        - **Visualization**: Matplotlib
        
        ### 🌟 Features
        
        - High-resolution (10m) Sentinel-2 imagery
        - Free access via Copernicus Data Space
        - AI-powered wildfire classification
        - Interactive region selection
        - Confidence score visualization
        - Support for custom image uploads
        """)
    
    with col2:
        st.markdown("""
        ### 📚 How It Works
        
        **Copernicus Sentinel-2 Mode:**
        1. Authenticates with Copernicus credentials
        2. Searches for recent Sentinel-2 L2A images
        3. Downloads high-resolution (10m) imagery
        4. Preprocesses image to 224×224 pixels
        5. Runs AI model for wildfire classification
        6. Returns prediction with confidence score
        
        **Upload Mode:**
        1. User uploads satellite or ground image
        2. Image is preprocessed to 224×224 pixels
        3. AI model analyzes the image
        4. Returns wildfire/no wildfire classification
        
        ### 🔗 Resources
        
        - [Copernicus Data Space](https://dataspace.copernicus.eu/)
        - [Get Free Credentials](https://dataspace.copernicus.eu/)
        - [Sentinel-2 Info](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-data/sentinel-2)
        
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
    
    **How Copernicus Works:**
    - OAuth2 authentication (secure)
    - Copernicus provides Sentinel-2 imagery free
    - 10m resolution perfect for wildfire detection
    - Global coverage with 5-day revisit time
    """)

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Built with ❤️ using Streamlit | Powered by Copernicus Sentinel-2 & TensorFlow</p>
        <p style="font-size: 0.9rem;">For educational and research purposes</p>
    </div>
    """,
    unsafe_allow_html=True
)
