import streamlit as st
import cv2
import numpy as np
import joblib
import os

model = joblib.load("weed_model.pkl")

st.set_page_config(page_title="Crop Health Monitor", layout="wide")

RGB_DIR = "data2019_miniscale/field_images/rgb"
NIR_DIR = "data2019_miniscale/field_images/nir"

st.title("🌾 AI-Powered Crop Health, Soil & Pest Risk Monitor")

available_files = os.listdir(RGB_DIR)
selected_file = st.selectbox("Select a field image to analyze:", available_files)

rgb_img = cv2.imread(os.path.join(RGB_DIR, selected_file))
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
nir_img = cv2.imread(os.path.join(NIR_DIR, selected_file), cv2.IMREAD_GRAYSCALE)

red_ch = rgb_img[:, :, 0].astype(float)
green_ch = rgb_img[:, :, 1].astype(float)
nir_ch = nir_img.astype(float)

ndvi = (nir_ch - red_ch) / (nir_ch + red_ch + 1e-6)
ndwi = (green_ch - nir_ch) / (green_ch + nir_ch + 1e-6)
brightness = rgb_img.astype(float)

# --- 11-feature extraction (must match training order exactly) ---
features = [[
    np.mean(ndvi),
    np.std(ndvi),
    np.median(ndvi),
    np.percentile(ndvi, 10),
    np.percentile(ndvi, 25),
    np.percentile(ndvi, 75),
    np.percentile(ndvi, 90),
    np.mean(ndwi),
    np.std(ndwi),
    np.mean(brightness),
    np.std(brightness),
]]

ndvi_mean = np.mean(ndvi)
ndvi_std = np.std(ndvi)
ndwi_mean = np.mean(ndwi)
brightness_mean = np.mean(brightness)

ml_prediction = model.predict(features)[0]
ml_confidence = model.predict_proba(features)[0][ml_prediction]

def classify_health(m):
    if m >= 0.4: return "🟢 Healthy"
    elif m >= 0.15: return "🟡 Moderate"
    else: return "🔴 Stressed"

def classify_soil(ndwi_m, bright_m):
    if ndwi_m > 0.1: return "💧 Moist/Wet"
    elif bright_m > 100 and ndwi_m < -0.1: return "🏜️ Dry/Bare"
    else: return "🟤 Normal"

def classify_pest(m, s):
    if s > 0.15 and m < 0.4: return "🔴 High Risk"
    elif s > 0.08: return "🟡 Moderate Risk"
    else: return "🟢 Low Risk"

col1, col2 = st.columns(2)
with col1:
    st.subheader("RGB Image")
    st.image(rgb_img, use_container_width=True)
with col2:
    st.subheader("NDVI Map")
    ndvi_display = ((ndvi + 1) / 2 * 255).astype(np.uint8)
    ndvi_colored = cv2.applyColorMap(ndvi_display, cv2.COLORMAP_RdYlGn if hasattr(cv2, 'COLORMAP_RdYlGn') else cv2.COLORMAP_JET)
    st.image(ndvi_colored, use_container_width=True)

st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Crop Health", classify_health(ndvi_mean), f"NDVI: {ndvi_mean:.3f}")
c2.metric("Soil Condition", classify_soil(ndwi_mean, brightness_mean), f"NDWI: {ndwi_mean:.3f}")
c3.metric("Pest Risk (rule-based)", classify_pest(ndvi_mean, ndvi_std), f"Patchiness: {ndvi_std:.3f}")

weed_label = "🐛 Weed Detected" if ml_prediction == 1 else "✅ No Weed"
c4.metric("ML Model Prediction", weed_label, f"Confidence: {ml_confidence:.1%}")

if ml_confidence < 0.65:
    st.warning("⚠️ Low confidence prediction — manual review recommended")