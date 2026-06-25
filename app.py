import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, UnidentifiedImageError

# ============================================================
# Konfigurasi aplikasi
# ============================================================
st.set_page_config(
    page_title="Deteksi Penyakit Kambing",
    page_icon="🐐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

IMG_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 10

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_PATH = MODEL_DIR / "best_efficientnetb0_kambing.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

DISPLAY_NAMES = {
    "orf": "Orf",
    "pink_eye": "Pink Eye",
    "scabies": "Scabies",
}

DISEASE_INFO = {
    "scabies": {
        "area": "Kulit dan area berbulu",
        "ciri": "Keropeng, luka, kulit menebal, kerontokan bulu, atau tekstur kulit tidak normal.",
        "saran": "Pisahkan sementara kambing yang dicurigai, jaga kebersihan kandang, lalu hubungi petugas kesehatan hewan."
    },
    "orf": {
        "area": "Mulut dan hidung",
        "ciri": "Lesi, kerak, benjolan, atau luka pada sekitar mulut dan hidung.",
        "saran": "Kurangi kontak dengan kambing lain, gunakan sarung tangan saat menangani, lalu konsultasikan ke dokter hewan."
    },
    "pink_eye": {
        "area": "Mata",
        "ciri": "Mata merah, berair, bengkak, kornea keruh, atau perubahan visual pada area mata.",
        "saran": "Hindari debu berlebih, pisahkan sementara bila perlu, lalu minta pemeriksaan dari tenaga kesehatan hewan."
    },
}

# ============================================================
# CSS sederhana: aman, jelas, dan mudah dibaca
# ============================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #FAFAF7;
    }
    .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #223322;
    }
    p, li, label, .stMarkdown {
        font-size: 18px;
        line-height: 1.55;
    }
    .big-note {
        padding: 18px 20px;
        border-radius: 14px;
        background: #F0F7EF;
        border: 1px solid #D5E8D4;
        font-size: 18px;
        line-height: 1.55;
    }
    .warning-note {
        padding: 18px 20px;
        border-radius: 14px;
        background: #FFF7E6;
        border: 1px solid #F0D59A;
        font-size: 18px;
        line-height: 1.55;
    }
    .danger-note {
        padding: 18px 20px;
        border-radius: 14px;
        background: #FFF0F0;
        border: 1px solid #E4A7A7;
        font-size: 18px;
        line-height: 1.55;
    }
    .result-card {
        padding: 22px 24px;
        border-radius: 18px;
        background: #FFFFFF;
        border: 1px solid #D9D9D9;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-top: 12px;
        margin-bottom: 14px;
    }
    .small-muted {
        color: #555;
        font-size: 15px;
    }
    div.stButton > button:first-child {
        width: 100%;
        height: 54px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Custom preprocessing layer agar model .keras dengan Lambda bisa dibaca
# ============================================================
@keras.utils.register_keras_serializable()
def efficientnet_preprocess_layer(x):
    return preprocess_input(x)

@st.cache_resource
def load_model_and_config():
    if not MODEL_PATH.exists():
        return None, None, None, f"Model tidak ditemukan di: {MODEL_PATH}"

    class_names = ["orf", "pink_eye", "scabies"]
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)

    threshold = 0.75
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "r") as f:
            threshold = float(json.load(f).get("confidence_threshold", threshold))

    custom_objects = {"efficientnet_preprocess_layer": efficientnet_preprocess_layer}

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False
        )
    except TypeError:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects
        )

    return model, class_names, threshold, None

def preprocess_uploaded_image(image: Image.Image):
    image_rgb = image.convert("RGB")
    resized = image_rgb.resize(IMG_SIZE)
    arr = np.array(resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr, image_rgb

def predict(model, class_names, image: Image.Image):
    arr, image_rgb = preprocess_uploaded_image(image)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_class, confidence, probs, image_rgb

def format_percent(x):
    return f"{x * 100:.2f}%"

# ============================================================
# Header
# ============================================================
st.title("🐐 Klasifikasi Penyakit Kambing")
st.markdown(
    """
    <div class="big-note">
    Aplikasi ini membantu mengenali kemungkinan penyakit <b>Scabies</b>, <b>Orf</b>, atau <b>Pink Eye</b> dari foto kambing.
    Hasil sistem adalah <b>alat bantu identifikasi awal</b>, bukan pengganti diagnosis dokter hewan.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ============================================================
# Load model
# ============================================================
model, class_names, default_threshold, error_message = load_model_and_config()

if error_message:
    st.markdown(
        f"""
        <div class="danger-note">
        <b>Model belum siap.</b><br>
        {error_message}<br><br>
        Salin file <code>best_efficientnetb0_kambing.keras</code>, <code>class_names.json</code>,
        dan <code>threshold.json</code> ke folder <code>model/</code>, lalu jalankan ulang aplikasi.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ============================================================
# Panduan penggunaan
# ============================================================
st.subheader("Cara Pakai")
st.markdown(
    """
    1. Pilih foto kambing yang jelas.
    2. Usahakan area gejala terlihat dekat dan tidak terlalu gelap.
    3. Klik tombol **Mulai Prediksi**.
    4. Baca hasil dan nilai keyakinan model.
    """
)

with st.expander("Pengaturan lanjutan"):
    threshold = st.slider(
        "Ambang keyakinan untuk hasil meyakinkan",
        min_value=0.50,
        max_value=0.95,
        value=float(default_threshold),
        step=0.01,
        help="Jika nilai keyakinan model di bawah ambang ini, sistem akan menampilkan peringatan hasil belum meyakinkan."
    )
else_threshold = float(default_threshold) if "threshold" not in locals() else threshold

uploaded_file = st.file_uploader(
    "Upload foto kambing",
    type=["jpg", "jpeg", "png", "webp"],
    help="Gunakan file gambar JPG, JPEG, PNG, atau WEBP. Maksimal 10 MB."
)

if uploaded_file is None:
    st.info("Silakan upload foto terlebih dahulu.")
    st.stop()

if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
    st.error(f"Ukuran file terlalu besar. Maksimal {MAX_FILE_SIZE_MB} MB.")
    st.stop()

try:
    image = Image.open(uploaded_file)
except UnidentifiedImageError:
    st.error("File tidak dapat dibaca sebagai gambar. Gunakan JPG, PNG, atau WEBP.")
    st.stop()

st.image(image, caption="Preview foto yang diunggah", use_container_width=True)

if st.button("Mulai Prediksi", type="primary"):
    with st.spinner("Sedang memproses gambar..."):
        pred_class, confidence, probs, image_rgb = predict(model, class_names, image)

    pred_display = DISPLAY_NAMES.get(pred_class, pred_class)
    confident = confidence >= else_threshold

    if confident:
        st.markdown(
            f"""
            <div class="result-card">
            <h2>Hasil Prediksi: {pred_display}</h2>
            <p><b>Keyakinan model:</b> {format_percent(confidence)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="warning-note">
            <h2>Hasil Belum Meyakinkan</h2>
            <p>Prediksi terdekat adalah <b>{pred_display}</b>, tetapi keyakinan model hanya <b>{format_percent(confidence)}</b>.</p>
            <p>Foto mungkin kurang jelas, bukan termasuk tiga penyakit utama, atau bisa saja kambing dalam kondisi sehat. Silakan gunakan foto lain atau lakukan pemeriksaan lanjutan.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Probabilitas Setiap Kelas")
    prob_df = pd.DataFrame({
        "Kelas": [DISPLAY_NAMES.get(c, c) for c in class_names],
        "Probabilitas": probs
    }).sort_values("Probabilitas", ascending=False)

    for _, row in prob_df.iterrows():
        st.write(f"**{row['Kelas']}** — {format_percent(float(row['Probabilitas']))}")
        st.progress(float(row["Probabilitas"]))

    st.subheader("Informasi Umum")
    info = DISEASE_INFO.get(pred_class)
    if info:
        st.markdown(
            f"""
            <div class="big-note">
            <b>Area gejala umum:</b> {info['area']}<br>
            <b>Ciri visual yang sering diamati:</b> {info['ciri']}<br>
            <b>Saran kehati-hatian:</b> {info['saran']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Catatan: Model hanya dilatih untuk tiga kelas penyakit. Untuk foto sehat atau penyakit lain, sistem mengandalkan nilai confidence untuk memberi peringatan hasil belum meyakinkan."
    )

st.write("")
st.markdown(
    """
    <p class="small-muted">
    Tips foto: gunakan pencahayaan cukup, fokuskan kamera ke area gejala, hindari gambar terlalu jauh,
    dan jangan gunakan foto yang sangat buram.
    </p>
    """,
    unsafe_allow_html=True,
)