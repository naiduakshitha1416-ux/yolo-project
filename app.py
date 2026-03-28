import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.title("PAN Card Detection")


MODEL_PATH = os.path.join(os.getcwd(), "best.pt")
model = YOLO(MODEL_PATH)


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
 
    results = model(image)    
    result_img = results[0].plot()

    st.image(result_img, caption="Detected Output", use_column_width=True)