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
/* HEADER */
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
/* GAMBAR */
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
/* TOMBOL ANALISIS */
.stButton > button {
    background-color: #E8F5E9 !important;
    color: #2E7D32 !important;
    border: 1px solid #A5D6A7 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #C8E6C9 !important;
    border: 1px solid #81C784 !important;
    color: #1B5E20 !important;
}
.stButton > button:focus {
    box-shadow: none !important;
}
/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}
/* METRIC */
[data-testid="stMetric"] {
    background: #FAFAFA;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #EEEEEE;
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
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    try:
        model.load_weights("rice_leaf_model_new.weights.h5")
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()

    labels = [
        "Bacterial Leaf Blight",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Not Leaf"
    ]
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
# UPLOAD & LOGIC
# ======================
uploaded_file = st.file_uploader(
    "📤 Pilih gambar daun padi untuk dianalisis",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG. Pastikan gambar jelas dan fokus pada daun padi."
)

# Jika user menghapus file/belum upload, reset session state agar tidak loop hasil lama
if uploaded_file is None:
    st.session_state.preds = None
    st.session_state.idx = None
    st.session_state.conf = None
    st.session_state.img_input = None
    st.session_state.file_name = None
else:
    # Jika ada file baru yang diupload (atau file berubah)
    if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        img_input = Image.open(uploaded_file).convert("RGB")
        
        # Jalankan prediksi langsung tanpa tombol tambahan untuk menghindari bug "button click rerun" di cloud
        with st.spinner("🔄 Menganalisis gambar..."):
            preds, idx, conf = predict_image(img_input, model)
        
        # Simpan ke session state
        st.session_state.preds = preds
        st.session_state.idx = idx
        st.session_state.conf = conf
        st.session_state.img_input = img_input
        st.session_state.file_name = uploaded_file.name

# ======================
# TAMPILKAN HASIL
# ======================
if st.session_state.get("preds") is not None:
    preds = st.session_state.preds
    idx = st.session_state.idx
    conf = st.session_state.conf
    img_input = st.session_state.img_input
    file_name = st.session_state.file_name

    col1, col2 = st.columns(2)

    # KIRI: Tampilan Gambar
    with col1:
        st.subheader("Gambar Input")
        buffered = BytesIO()
        img_input.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f'''
            <div class="figure-box">
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{img_str}" style="max-width:240px;border-radius:8px;">
                    <p style="margin-top:10px;font-weight:600;color:#333;">📄 {file_name}</p>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # KANAN: Hasil Analisis
    with col2:
        st.subheader("Hasil Analisis")

        if labels[idx] == "Healthy Rice Leaf":
            st.success(f"✅ Prediksi: {labels[idx]}")
        elif labels[idx] == "Not Leaf":
            st.warning("📷 Gambar yang diunggah bukan daun padi atau objek tidak dapat dikenali.")
        else:
            st.error(f"⚠️ Prediksi: {labels[idx]}")

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