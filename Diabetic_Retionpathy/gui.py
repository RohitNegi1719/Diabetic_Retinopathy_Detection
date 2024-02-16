import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
new_model = tf.keras.models.load_model("dr_model.h5")

st.title("Diabetic Retinopathy Detection")

uploaded_image = st.file_uploader("Upload an eye image...", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    RGBImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))

    st.image(RGBImg, caption="Uploaded Image", use_column_width=True)

    image = np.array(RGBImg) / 255.0
    predict = new_model.predict(np.array([image]))
    per = np.argmax(predict, axis=1)

    if per == 0:
        st.write("Prediction: No Diabetic Retinopathy")
    else:
        st.write("Prediction: Diabetic Retinopathy")