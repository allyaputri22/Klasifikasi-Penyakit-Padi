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
        print("Model berhasil dimuat")
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
# SESSION STATE
# ======================
if "preds" not in st.session_state:
    st.session_state.preds = None

if "idx" not in st.session_state:
    st.session_state.idx = None

if "conf" not in st.session_state:
    st.session_state.conf = None

if "img_input" not in st.session_state:
    st.session_state.img_input = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader(
    "📤 Pilih gambar daun padi untuk dianalisis",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG. Pastikan gambar jelas dan fokus pada daun padi."
)

if uploaded_file is not None:

    img_input = Image.open(uploaded_file).convert("RGB")

    if st.button("🔍 Mulai Analisis", type="primary", use_container_width=True):

        with st.spinner("🔄 Menganalisis gambar..."):
            preds, idx, conf = predict_image(img_input, model)

        # Simpan hasil ke session_state
        st.session_state.preds = preds
        st.session_state.idx = idx
        st.session_state.conf = conf
        st.session_state.img_input = img_input
        st.session_state.file_name = uploaded_file.name


# ======================
# TAMPILKAN HASIL
# ======================
if st.session_state.preds is not None:

    preds = st.session_state.preds
    idx = st.session_state.idx
    conf = st.session_state.conf
    img_input = st.session_state.img_input
    file_name = st.session_state.file_name

    col1, col2 = st.columns(2)

    # KIRI
    with col1:

        st.subheader("Gambar Input")

        buffered = BytesIO()
        img_input.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f'''
            <div class="figure-box">
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{img_str}"
                         style="max-width:240px;border-radius:8px;">
                    <p style="margin-top:10px;font-weight:600;color:#333;">
                        📄 {file_name}
                    </p>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # KANAN
    with col2:

        st.subheader("Hasil Analisis")

        if labels[idx] == "Healthy Rice Leaf":
            st.success(f"✅ Prediksi: {labels[idx]}")

        elif labels[idx] == "Not Leaf":
            st.warning(
                "📷 Gambar yang diunggah bukan daun padi atau objek tidak dapat dikenali sebagai daun padi."
            )

        else:
            st.error(f"⚠️ Prediksi: {labels[idx]}")

        st.metric(
            "Tingkat Kepercayaan",
            f"{conf:.2f}%"
        )

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