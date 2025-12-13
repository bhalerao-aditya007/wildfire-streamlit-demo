# 🔥 Wildfire Detection Web App

A deep learning–based web application that detects the presence of wildfires in images.  
Built using **TensorFlow** and **Streamlit**, this app allows users to upload an image and receive a prediction along with confidence scores.

---

## 🚀 Live Demo

👉 **Try the app here:**  
https://wildfire-app-demo-pxdbsx8tvysk54gpsnczo2.streamlit.app/

---

## 🧠 Model Overview

- Binary image classification:
  - **Wildfire**
  - **No Wildfire**
- Trained using a Convolutional Neural Network (CNN)
- Input image size: **224 × 224**
- Output: probability score + class prediction
- Model format: `.keras`

> The trained model is hosted on Google Drive and downloaded at runtime to avoid GitHub size limits.

---

## 🖼️ Features

- 📤 Image upload (JPG, JPEG, PNG, BMP, GIF)
- 🔍 Real-time inference
- 📊 Confidence score visualization
- ⚡ Lightweight, fast, and CPU-compatible
- ☁️ Deployed on Streamlit Cloud

---

## 🛠️ Tech Stack

- **Frontend / UI**: Streamlit  
- **Deep Learning**: TensorFlow / Keras  
- **Image Processing**: Pillow  
- **Visualization**: Matplotlib  
- **Model Hosting**: Google Drive (`gdown`)

---

## 📦 Installation (Local Setup)

Clone the repository:
```bash
git clone https://github.com/your-username/wildfire-detection-app.git
cd wildfire-detection-app
