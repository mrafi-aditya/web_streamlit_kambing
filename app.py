import hashlib
import html
import io
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================================
# Konfigurasi aplikasi
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Kondisi Kambing",
    page_icon="🐐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_MAX_FILE_SIZE_MB = 10
DEFAULT_ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
DEFAULT_CLASS_NAMES = ["orf", "pink_eye", "scabies", "sehat"]

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
MODEL_PATH = MODEL_DIR / "best_efficientnetb0_kambing.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
INPUT_CONFIG_PATH = MODEL_DIR / "input_config.json"
TEST_METRICS_PATH = MODEL_DIR / "test_metrics.json"


def canonicalize_class_name(value: str) -> str:
    """Menyeragamkan variasi penulisan nama kelas dari berkas JSON."""
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pinkeye": "pink_eye",
        "healthy": "sehat",
        "normal": "sehat",
        "healthy_goat": "sehat",
        "goat_healthy": "sehat",
    }
    return aliases.get(normalized, normalized)


def load_json_file(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


def load_input_settings():
    """Membaca konfigurasi input hasil notebook dengan nilai bawaan yang aman."""
    config = load_json_file(INPUT_CONFIG_PATH, {})
    if not isinstance(config, dict):
        config = {}

    try:
        width = int(config.get("image_width", DEFAULT_IMG_SIZE[0]))
        height = int(config.get("image_height", DEFAULT_IMG_SIZE[1]))
        if width <= 0 or height <= 0:
            raise ValueError
    except (TypeError, ValueError):
        width, height = DEFAULT_IMG_SIZE

    try:
        max_file_size_mb = int(
            config.get("max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB)
        )
        if max_file_size_mb <= 0:
            raise ValueError
    except (TypeError, ValueError):
        max_file_size_mb = DEFAULT_MAX_FILE_SIZE_MB

    raw_extensions = config.get(
        "allowed_extensions",
        DEFAULT_ALLOWED_EXTENSIONS,
    )
    if not isinstance(raw_extensions, list) or not raw_extensions:
        raw_extensions = DEFAULT_ALLOWED_EXTENSIONS

    allowed_extensions = []
    for extension in raw_extensions:
        extension = str(extension).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in allowed_extensions:
            allowed_extensions.append(extension)

    if not allowed_extensions:
        allowed_extensions = DEFAULT_ALLOWED_EXTENSIONS.copy()

    return {
        "image_size": (width, height),
        "max_file_size_mb": max_file_size_mb,
        "allowed_extensions": allowed_extensions,
        "upload_extensions": [ext.lstrip(".") for ext in allowed_extensions],
        "color_mode": str(config.get("color_mode", "RGB")).upper(),
        "class_order": config.get("class_order", DEFAULT_CLASS_NAMES),
    }


INPUT_SETTINGS = load_input_settings()
IMG_SIZE = INPUT_SETTINGS["image_size"]
MAX_FILE_SIZE_MB = INPUT_SETTINGS["max_file_size_mb"]
ALLOWED_EXTENSIONS = set(INPUT_SETTINGS["allowed_extensions"])
UPLOAD_EXTENSIONS = INPUT_SETTINGS["upload_extensions"]

DISPLAY_NAMES = {
    "orf": "Orf",
    "pink_eye": "Pink Eye",
    "scabies": "Scabies",
    "sehat": "Sehat",
}

CLASS_INFO = {
    "scabies": {
        "area": "Kulit dan bagian tubuh yang berbulu",
        "ciri": "Keropeng, luka, kulit menebal, bulu rontok, atau permukaan kulit tidak normal.",
        "saran": "Pisahkan sementara kambing, jaga kebersihan kandang, dan hubungi petugas kesehatan hewan.",
    },
    "orf": {
        "area": "Mulut dan hidung",
        "ciri": "Kerak, benjolan, luka, atau perubahan kulit di sekitar mulut dan hidung.",
        "saran": "Kurangi kontak dengan kambing lain, gunakan sarung tangan, dan hubungi petugas kesehatan hewan.",
    },
    "pink_eye": {
        "area": "Mata",
        "ciri": "Mata merah, berair, bengkak, keruh, atau tampak mengalami iritasi.",
        "saran": "Jauhkan kambing dari debu, pisahkan sementara bila perlu, dan hubungi petugas kesehatan hewan.",
    },
    "sehat": {
        "area": "Mulut atau hidung, mata, kulit, dan bulu yang terlihat pada foto",
        "ciri": "Tidak terlihat ciri visual utama Orf, Pink Eye, atau Scabies pada area yang diperiksa.",
        "saran": "Tetap pantau kondisi kambing. Hasil Sehat hanya berlaku pada pola visual di foto dan bukan pemeriksaan kesehatan menyeluruh.",
    },
}

# ============================================================
# Tampilan sederhana dan mudah digunakan
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --green-dark: #173D2B;
        --green: #246342;
        --green-soft: #EAF4EC;
        --cream: #F7F5EE;
        --white: #FFFFFF;
        --text: #1D2A22;
        --muted: #5B685F;
        --border: #CBD7CE;
        --yellow-soft: #FFF3D4;
        --yellow-border: #E2BD5A;
        --red-soft: #FDECEC;
        --red-border: #E0AAAA;
    }

    #MainMenu,
    footer,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }

    .stApp {
        background: var(--cream);
        color: var(--text);
    }

    .block-container {
        max-width: 900px;
        padding: 0.75rem 1rem 1.25rem;
    }

    .app-header {
        background: var(--green-dark);
        border-radius: 20px;
        padding: 18px 22px;
        margin-bottom: 14px;
        color: white;
        box-shadow: 0 8px 24px rgba(23, 61, 43, 0.15);
    }

    .app-title {
        font-size: clamp(1.65rem, 4vw, 2.3rem);
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .app-subtitle {
        color: #E6EFE9;
        font-size: 16px;
        line-height: 1.4;
    }

    .main-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(30, 70, 45, 0.08);
        margin-bottom: 14px;
    }

    .section-title {
        color: var(--green-dark);
        font-size: 22px;
        font-weight: 900;
        margin-bottom: 5px;
    }

    .section-help {
        color: var(--muted);
        font-size: 16px;
        line-height: 1.45;
    }

    .tips {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 13px;
    }

    .tip {
        background: var(--green-soft);
        border: 1px solid #C8DECD;
        border-radius: 12px;
        padding: 10px;
        color: var(--green-dark);
        font-size: 14px;
        font-weight: 750;
        text-align: center;
    }

    div[data-testid="stCameraInput"],
    div[data-testid="stFileUploader"] {
        background: #FAFCFA;
        border: 2px dashed #A4B8A8;
        border-radius: 16px;
        padding: 10px;
    }

    div[data-testid="stCameraInput"] button,
    div[data-testid="stFileUploader"] button {
        min-height: 50px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 800;
    }

    div[role="radiogroup"] {
        gap: 8px;
    }

    div[role="radiogroup"] label {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 9px 13px;
    }

    div.stButton > button {
        width: 100%;
        min-height: 56px;
        border-radius: 14px;
        font-size: 19px;
        font-weight: 900;
        border: 2px solid var(--green);
    }

    div.stButton > button[kind="primary"] {
        background: var(--green);
        color: white;
    }

    div[data-testid="stImage"] img {
        max-height: 340px;
        object-fit: contain;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: #F4F6F4;
    }

    .result-card {
        height: 100%;
        background: linear-gradient(145deg, #E7F4EA, #FAFDFA);
        border: 2px solid #9FC4A8;
        border-radius: 18px;
        padding: 20px;
    }

    .result-card.low {
        background: linear-gradient(145deg, #FFF0C8, #FFFBF1);
        border-color: var(--yellow-border);
    }

    .result-label {
        color: var(--muted);
        font-size: 14px;
        font-weight: 850;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .result-name {
        color: var(--green-dark);
        font-size: clamp(2.3rem, 7vw, 3.5rem);
        font-weight: 950;
        line-height: 1;
        margin: 10px 0;
    }

    .confidence {
        color: #314039;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .info-box {
        background: white;
        border: 1px solid var(--border);
        border-left: 5px solid var(--green);
        border-radius: 13px;
        padding: 12px 14px;
        margin-top: 10px;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    .info-box-title {
        color: var(--green-dark);
        font-size: 15px;
        font-weight: 900;
        line-height: 1.35;
        margin-bottom: 4px;
    }

    .info-box-text {
        color: #34433A;
        font-size: 15px;
        line-height: 1.5;
    }

    .warning-box {
        background: var(--yellow-soft);
        border: 1px solid var(--yellow-border);
        border-radius: 14px;
        padding: 13px 15px;
        margin-top: 12px;
        color: #634200;
        font-size: 15px;
        line-height: 1.45;
    }

    .danger-box {
        background: var(--red-soft);
        border: 1px solid var(--red-border);
        border-radius: 14px;
        padding: 13px 15px;
        margin-top: 14px;
        color: #722525;
        font-size: 14px;
        line-height: 1.45;
    }

    .prob-row {
        margin: 11px 0;
    }

    .prob-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 15px;
        font-weight: 800;
    }

    .prob-track {
        height: 12px;
        background: #E3E9E4;
        border-radius: 999px;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        background: var(--green);
        border-radius: 999px;
    }

    .footer-note {
        color: var(--muted);
        text-align: center;
        font-size: 13px;
        margin-top: 10px;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 0.5rem 0.75rem 1rem;
        }

        .app-header,
        .main-card {
            padding: 16px;
            border-radius: 17px;
        }

        .tips {
            grid-template-columns: 1fr;
        }

        .tip {
            text-align: left;
        }

        div[data-testid="stImage"] img {
            max-height: 300px;
        }

        div.stButton > button {
            min-height: 54px;
            font-size: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model dan pemrosesan gambar
# ============================================================
@keras.utils.register_keras_serializable()
def efficientnet_preprocess_layer(x):
    return preprocess_input(x)


@st.cache_resource(show_spinner=False)
def load_model_and_config():
    if not MODEL_PATH.exists():
        return None, None, None, f"Model tidak ditemukan di: {MODEL_PATH}"

    class_names = DEFAULT_CLASS_NAMES.copy()

    loaded_names = load_json_file(CLASS_NAMES_PATH, None)
    if isinstance(loaded_names, list) and loaded_names:
        class_names = [canonicalize_class_name(name) for name in loaded_names]
    else:
        config_order = INPUT_SETTINGS.get("class_order")
        if isinstance(config_order, list) and config_order:
            class_names = [canonicalize_class_name(name) for name in config_order]

    if len(class_names) != len(set(class_names)):
        return (
            None,
            None,
            None,
            "Urutan kelas pada class_names.json memuat nama kelas yang berulang.",
        )

    threshold = 0.75
    threshold_data = load_json_file(THRESHOLD_PATH, threshold)
    try:
        if isinstance(threshold_data, dict):
            threshold = float(
                threshold_data.get("confidence_threshold", threshold)
            )
        else:
            threshold = float(threshold_data)
    except (ValueError, TypeError):
        threshold = 0.75

    threshold = max(0.0, min(threshold, 1.0))

    custom_objects = {
        "efficientnet_preprocess_layer": efficientnet_preprocess_layer
    }

    try:
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False,
            )
        except TypeError:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects=custom_objects,
            )
    except Exception as error:
        return None, None, None, f"Model gagal dibuka: {error}"

    try:
        output_shape = model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        model_class_count = int(output_shape[-1])
    except (TypeError, ValueError, IndexError):
        model_class_count = None

    if model_class_count is not None and model_class_count != len(class_names):
        return (
            None,
            None,
            None,
            "Jumlah keluaran model tidak sesuai dengan class_names.json: "
            f"model={model_class_count}, kelas={len(class_names)}.",
        )

    return model, class_names, threshold, None

def validate_file_extension(file_name: str):
    extension = Path(file_name).suffix.lower()
    if extension and extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(ext.upper().lstrip(".") for ext in sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Format foto tidak didukung. Gunakan {supported}."
        )


def read_image(file_bytes: bytes) -> Image.Image:
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"Ukuran foto terlalu besar. Maksimal {MAX_FILE_SIZE_MB} MB."
        )

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "Foto tidak dapat dibaca. Gunakan JPG, JPEG, PNG, atau WEBP."
        ) from error

    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(image: Image.Image):
    resized = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype="float32")
    return np.expand_dims(image_array, axis=0)


def predict_image(model, class_names, image: Image.Image):
    image_array = preprocess_image(image)
    probabilities = np.asarray(
        model.predict(image_array, verbose=0)[0],
        dtype="float32",
    ).reshape(-1)

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Jumlah probabilitas keluaran model tidak sesuai dengan urutan kelas."
        )

    prediction_index = int(np.argmax(probabilities))
    prediction_class = class_names[prediction_index]
    confidence = float(probabilities[prediction_index])
    return prediction_class, confidence, probabilities

def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


# ============================================================
# Session state
# ============================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "active_photo_hash" not in st.session_state:
    st.session_state.active_photo_hash = None


# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🐐 Klasifikasi Kondisi Kambing</div>
        <div class="app-subtitle">
            Masukkan foto, tekan <b>Periksa Foto</b>, lalu hasil akan tampil bersama fotonya.
            Model mengklasifikasikan empat kelas: <b>Orf, Pink Eye, Scabies,</b> dan <b>Sehat</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Memuat model
# ============================================================
with st.spinner("Menyiapkan sistem..."):
    model, class_names, confidence_threshold, model_error = load_model_and_config()

if model_error:
    st.error(model_error)
    st.info(
        "Pastikan file model berada dalam folder model/ bersama file "
        "class_names.json, threshold.json, dan input_config.json."
    )
    st.stop()


# ============================================================
# Input foto
# ============================================================
st.markdown(
    """
    <div class="main-card">
        <div class="section-title">Masukkan foto kambing</div>
        <div class="section-help">
            Foto harus fokus pada bagian tubuh yang ingin diperiksa.
        </div>
        <div class="tips">
            <div class="tip">☀️ Gunakan cahaya yang cukup</div>
            <div class="tip">🎯 Area pemeriksaan terlihat jelas</div>
            <div class="tip">🔎 Ambil dari jarak dekat</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

photo_source = st.radio(
    "Sumber foto",
    ["📷 Kamera langsung", "🖼️ Pilih dari galeri"],
    horizontal=True,
    label_visibility="collapsed",
)

photo_file = None

if photo_source == "📷 Kamera langsung":
    photo_file = st.camera_input(
        "Ambil foto kambing",
        help="Izinkan browser menggunakan kamera ketika diminta.",
    )
else:
    photo_file = st.file_uploader(
        "Pilih foto kambing",
        type=UPLOAD_EXTENSIONS,
        help=f"Ukuran maksimal {MAX_FILE_SIZE_MB} MB.",
    )


# ============================================================
# Validasi dan preview foto
# ============================================================
photo = None
photo_hash = None

if photo_file is not None:
    try:
        validate_file_extension(getattr(photo_file, "name", ""))
        photo_bytes = photo_file.getvalue()
        photo = read_image(photo_bytes)
        photo_hash = hashlib.sha256(photo_bytes).hexdigest()
    except ValueError as error:
        st.error(str(error))

if photo_hash != st.session_state.active_photo_hash:
    st.session_state.active_photo_hash = photo_hash
    st.session_state.prediction_result = None


# ============================================================
# Belum ada foto
# ============================================================
if photo is None:
    st.info("Silakan ambil atau pilih foto terlebih dahulu.")
    st.markdown(
        """
        <div class="footer-note">
            Hasil aplikasi merupakan klasifikasi awal dan bukan diagnosis dokter hewan.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# Tombol periksa
# ============================================================
if st.session_state.prediction_result is None:
    st.markdown(
        """
        <div class="main-card">
            <div class="section-title">Foto yang akan diperiksa</div>
            <div class="section-help">
                Pastikan foto tidak buram dan area yang diperiksa terlihat jelas.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(photo, use_container_width=True)

    if st.button("🔍 Periksa Foto", type="primary"):
        try:
            with st.spinner("Sedang memeriksa foto..."):
                prediction_class, confidence, probabilities = predict_image(
                    model,
                    class_names,
                    photo,
                )

            st.session_state.prediction_result = {
                "photo_hash": photo_hash,
                "prediction_class": prediction_class,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
            }
            st.rerun()
        except Exception as error:
            st.error(f"Prediksi gagal dilakukan: {error}")


# ============================================================
# Hasil beserta gambar
# ============================================================
result = st.session_state.prediction_result

if result is not None and result.get("photo_hash") == photo_hash:
    prediction_class = result["prediction_class"]
    confidence = float(result["confidence"])
    probabilities = np.asarray(result["probabilities"], dtype="float32")
    prediction_name = DISPLAY_NAMES.get(
        prediction_class,
        prediction_class,
    )
    info = CLASS_INFO.get(prediction_class)
    is_confident = confidence >= confidence_threshold
    result_class = "result-card" if is_confident else "result-card low"

    st.markdown(
        """
        <div class="main-card">
            <div class="section-title">Hasil pemeriksaan foto</div>
            <div class="section-help">
                Foto dan hasil klasifikasi ditampilkan bersama agar mudah diperiksa kembali.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_column, result_column = st.columns([1.05, 1])

    with image_column:
        st.image(
            photo,
            caption="Foto yang diperiksa",
            use_container_width=True,
        )

    with result_column:
        # Bagian hasil dibuat terpisah agar HTML tidak terbaca sebagai blok kode.
        st.markdown(
            (
                f'<div class="{result_class}">'
                f'<div class="result-label">Hasil klasifikasi kondisi</div>'
                f'<div class="result-name">{html.escape(prediction_name)}</div>'
                f'<div class="confidence">'
                f'Keyakinan model: {format_percent(confidence)}'
                f'</div>'
                f'</div>'
            ),
            unsafe_allow_html=True,
        )

        if not is_confident:
            st.markdown(
                (
                    '<div class="warning-box">'
                    '<b>Hasil belum cukup kuat.</b><br>'
                    'Model tetap memilih salah satu dari empat kelas. '
                    'Coba gunakan foto yang lebih dekat, terang, tidak buram, dan memperlihatkan area pemeriksaan dengan jelas.'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

        if info:
            if prediction_class == "sehat":
                info_items = [
                    ("📍", "Area yang diperiksa", info["area"]),
                    ("🔎", "Interpretasi visual", info["ciri"]),
                    ("✅", "Catatan", info["saran"]),
                ]
            else:
                info_items = [
                    ("📍", "Bagian yang biasanya terdampak", info["area"]),
                    ("🔎", "Ciri umum yang terlihat", info["ciri"]),
                    ("✅", "Saran awal", info["saran"]),
                ]

            for icon, title, description in info_items:
                st.markdown(
                    (
                        '<div class="info-box">'
                        f'<div class="info-box-title">{icon} {html.escape(title)}</div>'
                        f'<div class="info-box-text">{html.escape(description)}</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

    with st.expander("Lihat rincian probabilitas setiap kelas"):
        ordered_indices = np.argsort(probabilities)[::-1]

        for index in ordered_indices:
            class_name = class_names[int(index)]
            probability = float(probabilities[int(index)])
            display_name = DISPLAY_NAMES.get(class_name, class_name)
            width = max(0.0, min(probability * 100.0, 100.0))

            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-head">
                        <span>{display_name}</span>
                        <span>{format_percent(probability)}</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{width:.1f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="danger-box">
            <b>Penting:</b> model hanya membedakan Orf, Pink Eye, Scabies, dan Sehat.
            Hasil Sehat berarti ciri visual utama ketiga penyakit tidak terlihat pada
            area di dalam foto, bukan jaminan bahwa seluruh kondisi kesehatan kambing
            normal. Hasil aplikasi tidak menggantikan pemeriksaan dokter hewan.
        </div>
        """,
        unsafe_allow_html=True,
    )

