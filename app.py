import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Padi Sehat AI - Klasifikasi Penyakit Daun Padi",
    layout="wide"
)

# ======================
# STYLE (MODERN UI)
# ======================
st.markdown("""
<style>
.block-container {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 2rem;
}

.main-header {
    text-align: center;
    color: white;
    background: linear-gradient(135deg, #1B5E20, #43A047);
    padding: 30px;
    border-radius: 15px;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.sub-text {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.upload-section {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #4CAF50;
    margin-bottom: 20px;
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

    .figure-box {
        background: #ffffff;
        border: 1px solid #dfe6e9;
        border-radius: 18px;
        padding: 18px;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 16px;
        min-height: 420px;
    }

    .figure-box img {
        border-radius: 14px;
        max-width: 100%;
        height: auto;
        object-fit: cover;
        display: block;
    }

.prob-section {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Padi Sehat AI - Klasifikasi Penyakit Daun Padi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Ketahui jenis penyakit daun padi anda hanya dengan satu kali klik</div>', unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model_rebuild():
    base_model = tf.keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    try:
        model.load_weights("rice_leaf_model.weights.h5")
        print("Model berhasil dimuat")
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()

    labels = ["Bacterial Leaf Blight", "Healthy Rice Leaf", "Leaf Blast"]
    return model, labels


model, labels = load_model_rebuild()

# ======================
# PREDICT FUNCTION
# ======================
def predict_image(image, model):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(preds)

    return preds, idx, float(preds[idx] * 100)

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader(
    "📤 Pilih gambar daun padi untuk dianalisis",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG. Pastikan gambar jelas dan fokus pada daun padi."
)

if uploaded_file:
    img_input = Image.open(uploaded_file).convert("RGB")

    if st.button("🔍 Mulai Analisis", type="primary", use_container_width=True):
        with st.spinner("🔄 Menganalisis gambar..."):
            preds, idx, conf = predict_image(img_input, model)

        # ======================
        # HASIL ANALISIS - 2 KOLOM
        # ======================
        col1, col2 = st.columns(2)

        # KIRI: Gambar
        with col1:
            st.subheader("Gambar Input")
            # Convert gambar ke base64
            buffered = BytesIO()
            img_input.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(f'<div class="figure-box"><img src="data:image/png;base64,{img_str}" style="max-width:240px; border-radius:8px;"></div>', unsafe_allow_html=True)

        # KANAN: Hasil
        with col2:
            st.subheader("Hasil Analisis")

            if labels[idx] == "Healthy Rice Leaf":
                st.success(f"✅ **Prediksi: {labels[idx]}**")
            else:
                st.error(f"⚠️ **Prediksi: {labels[idx]}**")
            st.metric("Tingkat Kepercayaan", f"{conf:.2f}%")

            st.markdown("### Probabilitas Kelas")
            for i, label in enumerate(labels):
                prob_percent = preds[i] * 100
                st.write(f"**{label}**: {prob_percent:.2f}%")
                st.progress(float(preds[i]))

# ======================
# FOOTER
# ======================
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px; margin-top: 50px;">
    <p>🌾 <strong>Padi Sehat AI</strong> - Membantu mengidentifikasi jenis penyakit padi secara cepat dan akurat.</p>
    <p>Copyright © 2026. All rights reserved. Developed with ❤️ by Allya Putri.</p>
</div>
""", unsafe_allow_html=True)