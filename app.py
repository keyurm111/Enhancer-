import streamlit as st
from PIL import Image
import numpy as np
import cv2
from RealESRGAN import RealESRGAN
import torch
import os
import io

# Page Config
st.set_page_config(page_title="AI Image Studio", page_icon="🖼️")

# Sidebar Navigation
mode = st.sidebar.selectbox("Choose Tool", ["AI Image Upscaler 🚀", "WebP Optimizer 📉"])

# Load Model
@st.cache_resource
def load_model():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth')
    return model, device.type

if mode == "AI Image Upscaler 🚀":
    st.title("🖼️ AI Image Upscaler")
    
    model, device = load_model()
    st.sidebar.info(f"Running on: **{device.upper()}**")
    if device == 'cpu':
        st.sidebar.warning("⚠️ CPU mode is slow. Larger images may take several minutes.")

    uploaded_file = st.file_uploader("Upload Image to Upscale", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Original Image")
        st.image(image, width="stretch")

        img_np = np.array(image)

        if st.button("Upscale Image 🚀"):
            with st.spinner("AI is enhancing your image..."):
                upscaled = model.predict(img_np)

            st.subheader("Upscaled Image")
            st.image(upscaled, width="stretch")

            # Save to buffer
            buf = io.BytesIO()
            upscaled.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Upscaled PNG",
                data=byte_im,
                file_name="upscaled.png",
                mime="image/png"
            )

else:
    st.title("📉 WebP Optimizer")
    st.write("Convert images to WebP to reduce size by up to 90%+ while maintaining quality.")
    
    uploaded_file = st.file_uploader("Upload Image to Compress", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.subheader("Preview")
        st.image(image, width="stretch")
        
        # Original size
        original_size = len(uploaded_file.getvalue()) / 1024
        st.write(f"Original Size: **{original_size:.2f} KB**")
        
        quality = st.slider("Optimization Quality (Higher = Better Quality)", 1, 100, 80)
        
        if st.button("Optimize Image ✨"):
            with st.spinner("Compressing..."):
                buf = io.BytesIO()
                image.save(buf, format="WEBP", quality=quality, method=6) # method 6 is slowest but best compression
                optimized_data = buf.getvalue()
                new_size = len(optimized_data) / 1024
                reduction = 100 - (new_size / original_size * 100)
                
            st.success(f"Optimized! New Size: **{new_size:.2f} KB** ({reduction:.1f}% reduction)")
            
            st.download_button(
                label="Download WebP",
                data=optimized_data,
                file_name="optimized.webp",
                mime="image/webp"
            )