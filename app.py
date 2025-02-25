import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Load model yang sudah disimpan
model = load_model("models.h5", compile=False)

# Load labels dari file JSON
with open("labels.json", "r") as f:
    class_indices = json.load(f)

# Balik mapping {label: index} menjadi {index: label}
labels = {v: k for k, v in class_indices.items()}

# Fungsi untuk memproses gambar sebelum prediksi
def preprocess_image(img):
    img = img.resize((224, 224))  # Sesuaikan ukuran dengan model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalisasi
    return img

# Streamlit UI
st.title("🔍 Klasifikasi Rempah-Rempah dengan CNN")
st.write("Upload gambar rempah-rempah dan model akan memprediksi jenisnya!")

uploaded_file = st.file_uploader("Unggah gambar rempah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # Prediksi saat tombol ditekan
    if st.button("Prediksi"):
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = labels[predicted_class]

        st.success(f"✅ Prediksi: {predicted_label}")
