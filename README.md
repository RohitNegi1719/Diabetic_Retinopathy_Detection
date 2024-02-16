# Diabetic Retinopathy Detection App

This repository contains the code for a web application that detects diabetic retinopathy from eye images using a deep learning model. The application is built with Streamlit, TensorFlow, and OpenCV, offering a user-friendly interface for uploading images and receiving predictions.

## Features

- **Image Upload**: Users can upload eye images in PNG, JPG, or JPEG formats.
- **Real-Time Prediction**: Instantly predicts the presence of diabetic retinopathy in the uploaded image.
- **Visual Feedback**: Displays the uploaded image and prediction result within the app.

## Usage
After installation, you can run the application using Streamlit:

streamlit run app.py

## Project Structure
- **dr_model.h5**: The TensorFlow/Keras model trained to detect diabetic retinopathy.
- **gui.py**: The Streamlit application script for the web interface.
- **main.py**: Code for model.

## Dataset
The model was trained on a dataset not included in this repository. You can find similar datasets from medical image databases or research institutes specializing in ophthalmology.
