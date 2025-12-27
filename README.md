# 🔥 Wildfire Detection Web App

An AI-powered web application for **early wildfire detection** using **satellite imagery** and **deep learning**.

Built in response to the growing impact of large-scale wildfires, highlighted by the **2023 Canadian wildfire season**, which caused severe air-quality emergencies across North America.

---

## 🌍 Motivation

In **2023**, wildfire smoke from Canada traveled over **800 km**, turning skies orange and pushing cities like **New York** to some of the worst air-quality levels in the world.

Wildfires don’t announce themselves.  
By the time flames are visible, response time is already limited.

This project focuses on **early, reliable wildfire detection** using AI.

---

## 🚀 Live Demo

👉 https://wildfire-app-demo-pxdbsx8tvysk54gpsnczo2.streamlit.app/

---

## 🧠 Model Overview

- **Task**: Binary image classification  
  - Wildfire  
  - No Wildfire
- **Architecture**: EfficientNetB4 (transfer learning)
- **Input**: 224 × 224 satellite images
- **Output**: Probability score + prediction
- **Model format**: `.keras`

---

## 📊 Performance

- **Accuracy**: 95.22%  
- **AUC-ROC**: 0.989  

---

## 🖼️ App Features

- 📤 Image upload  
- 🔍 Real-time inference  
- 📊 Confidence-based risk output  
- ⚡ Fast, CPU-compatible  
- ☁️ Deployed on Streamlit Cloud  

---

## 🛠️ Tech Stack

- TensorFlow / Keras  
- EfficientNetB4  
- Sentinel-2 satellite imagery  
- Google Earth Engine  
- Streamlit  
- Python, NumPy, Matplotlib  

---

## 📦 Local Setup

```bash
git clone https://github.com/bhalerao-aditya007/wildfire-streamlit-demo
cd wildfire-streamlit-demo
pip install -r requirements.txt
streamlit run app.py
