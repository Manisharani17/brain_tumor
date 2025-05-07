import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Settings
IMG_SIZE = 128
DATA_PATH = "brain_tumor_dataset"
# Load and preprocess images
@st.cache_data
def load_images(data_path, img_size=128):
    images = []
    labels = []
    for label, folder in enumerate(["no", "yes"]):
        folder_path = os.path.join(data_path, folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Train model (only once)
@st.cache_resource
def train_model():
    X, y = load_images(DATA_PATH)
    X = X / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0)
    return model

model = train_model()

# Streamlit UI
st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI image to check for a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    prediction = model.predict(img_input)[0][0]
    label = "Tumor" if prediction > 0.5 else "No Tumor"

    st.image(img, caption=f"Prediction: {label} ({prediction:.2f})", width=250)


