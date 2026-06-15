import io
import zipfile
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# streamlit-webrtc (live webcam)
try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# CONFIG
MODEL_PATH = Path("best_model_dinov2.pth")
IMG_SIZE   = 224
DB_PATH    = "predictions.db"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES_2 = ["crack", "spot"]
CLASSES_3 = ["crack", "pinhole", "spot"]

# Default pipeline params
DEFAULT_EXPOSURE      = 0.10   # alpha untuk convertScaleAbs
DEFAULT_SHARPEN_AMT   = 3.0   # weight untuk addWeighted
DEFAULT_CONF_THRESH   = 95.0  # %
DEFECT_CLASSES_FOR_THRESHOLD = ["crack", "spot", "pinhole"]

# White plate detection
WHITE_PLATE_HSV_MIN = np.array([0, 0, 180])
WHITE_PLATE_HSV_MAX = np.array([180, 50, 255])
WHITE_PLATE_RATIO_THRESHOLD = 0.65

# Auto-save cooldown (detik) agar DB tidak membludak saat live
AUTOSAVE_COOLDOWN = 5.0

# Max FPS untuk VideoProcessor (batasi beban CPU/GPU)
MAX_FPS = 10
_MIN_FRAME_INTERVAL = 1.0 / MAX_FPS  # ~0.1 s

# Export folder for scanned plates
SCANNED_DIR = Path("scannedplate")

# Live defect status variable: 1 = defect (crack/spot) , 0 = mulus
LIVE_DEFECT = 0

# Export cooldown to avoid writing every frame
EXPORT_COOLDOWN = 2.0

# DATABASE
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        source       TEXT    NOT NULL,
        filename     TEXT,
        label        TEXT    NOT NULL,
        confidence   REAL    NOT NULL,
        circle_found INTEGER DEFAULT 0,
        exposure     REAL    DEFAULT 1.3,
        sharpen      REAL    DEFAULT 1.5,
        created_at   TEXT    NOT NULL,
        image_data   BLOB
    )
    """)
    # Migrasi: tambah kolom baru jika belum ada (backward compat)
    for col_def in [
        "ALTER TABLE predictions ADD COLUMN exposure REAL DEFAULT 1.3",
        "ALTER TABLE predictions ADD COLUMN sharpen  REAL DEFAULT 1.5",
        "ALTER TABLE predictions ADD COLUMN image_data BLOB",
    ]:
        try:
            conn.execute(col_def)
        except Exception:
            pass
    conn.commit()
    conn.close()

def encode_image_bytes(img_rgb: np.ndarray, quality: int = 75) -> bytes:
    """Encode numpy RGB array ke JPEG bytes untuk disimpan di SQLite."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def save_prediction(source: str, filename: str, label: str,
                    confidence: float, circle_found: bool = False,
                    exposure: float = 1.3, sharpen: float = 1.5,
                    image_data: bytes | None = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions "
        "(source, filename, label, confidence, circle_found, exposure, sharpen, created_at, image_data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source, filename or "-", label, round(confidence, 4),
         int(circle_found), round(exposure, 2), round(sharpen, 2),
         datetime.now().isoformat(timespec="seconds"), image_data),
    )
    conn.commit()
    conn.close()

def get_history(limit: int = 50) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, source, filename, label, confidence, circle_found, exposure, sharpen, created_at, image_data "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows

def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    by_label = conn.execute(
        "SELECT label, COUNT(*) FROM predictions GROUP BY label ORDER BY COUNT(*) DESC "
    ).fetchall()
    avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    conn.close()
    return {"total": total, "by_label": by_label, "avg_conf": avg_conf or 0.0}

def clear_history() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

# MODEL — auto-detect num_classes dari checkpoint
def _get_classes_from_checkpoint(path: Path) -> list[str]:
    sd = torch.load(path, map_location="cpu")
    head_key = None
    for key in sd.keys():
        if key.endswith(".weight") and "head" in key:
            head_key = key
            break
    if head_key is None:
        for key in reversed(list(sd.keys())):
            if key.endswith(".weight") and sd[key].ndim == 2 and sd[key].shape[0] <= 10:
                head_key = key
                break
    if head_key is None:
        st.error("Tidak dapat menentukan jumlah kelas dari checkpoint.")
        st.stop()
    n = sd[head_key].shape[0]
    if n == 2:
        return CLASSES_2
    elif n == 3:
        return CLASSES_3
    else:
        return [f"class_{i}" for i in range(n)]

@st.cache_resource(show_spinner="Memuat model DINOv2...")
def load_model() -> tuple[nn.Module, list[str]]:
    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    classes = _get_classes_from_checkpoint(MODEL_PATH)
    num_classes = len(classes)
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        img_size=IMG_SIZE,
    )
    in_features = getattr(model, "num_features", None) or getattr(model, "embed_dim", None)
    model.head = nn.Linear(in_features, num_classes)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model, classes

# CIRCLE DETECTION
def detect_circle_mask(gray: np.ndarray) -> tuple[np.ndarray | None, bool, tuple | None]:
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min(h, w) * 0.4,
        param1=100,
        param2=40,
        minRadius=int(min(h, w) * 0.2),
        maxRadius=int(min(h, w) * 0.55),
    )
    if circles is None:
        return None, False, None
    circles = np.round(circles[0]).astype(int)
    cx, cy, r = max(circles, key=lambda c: c[2])
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
    return mask, True, (cx, cy, r)

def draw_circle_overlay(img_rgb: np.ndarray, cx: int, cy: int, r: int,
                        color: tuple = (0, 220, 0), thickness: int = 3) -> np.ndarray:
    overlay = img_rgb.copy()
    cv2.circle(overlay, (cx, cy), r, color, thickness)
    cv2.circle(overlay, (cx, cy), 5, color, -1)
    return overlay

def detect_white_plate(img_rgb: np.ndarray, mask: np.ndarray | None = None,
                       min_ratio: float = WHITE_PLATE_RATIO_THRESHOLD) -> tuple[bool, float]:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    white_mask = cv2.inRange(hsv, WHITE_PLATE_HSV_MIN, WHITE_PLATE_HSV_MAX)
    if mask is not None:
        white_mask = cv2.bitwise_and(white_mask, white_mask, mask=mask)
        total_pixels = np.count_nonzero(mask)
    else:
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    if total_pixels == 0:
        return False, 0.0

    white_ratio = np.count_nonzero(white_mask) / total_pixels
    return white_ratio >= min_ratio, float(white_ratio * 100.0)

# PREPROCESSING PIPELINE (exposure & sharpen bisa dikontrol)
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(img_rgb: np.ndarray,
               exposure: float = DEFAULT_EXPOSURE,
               sharpen_amt: float = DEFAULT_SHARPEN_AMT) -> dict:
    # 1. Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # 2. Deteksi lingkaran
    mask, circle_found, circle_info = detect_circle_mask(gray)
    # 3. Inversi Warna
    inverted = cv2.bitwise_not(gray)
    # 4. Exposure (alpha = exposure factor)
    exposed = cv2.convertScaleAbs(inverted, alpha=exposure, beta=0)
    # 5. Sharpen via Unsharp Mask
    blurred = cv2.GaussianBlur(exposed, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(exposed, sharpen_amt, blurred, -(sharpen_amt - 1), 0)
    # 6. Crop (apply mask setelah sharpen)
    if circle_found and mask is not None:
        cropped = cv2.bitwise_and(sharpened, sharpened, mask=mask)
    else:
        cropped = sharpened.copy()
    # 7. 3-channel
    rgb_3ch = cv2.merge([cropped, cropped, cropped])
    # 8. Transform ke tensor
    tensor = _transform(rgb_3ch).unsqueeze(0).to(DEVICE)
    # 9. Overlay lingkaran di gambar asli (display)
    annotated = img_rgb.copy()
    if circle_found and circle_info:
        cx, cy, r = circle_info
        annotated = draw_circle_overlay(img_rgb, cx, cy, r)

    white_plate = False
    white_ratio = 0.0
    if circle_found and mask is not None:
        # Hanya hitung area piring yang terdeteksi agar background tidak mempengaruhi.
        white_plate, white_ratio = detect_white_plate(img_rgb, mask=mask)

    return {
        "tensor": tensor,
        "gray": gray,
        "inverted": inverted,
        "exposed": exposed,
        "sharpened": sharpened,
        "cropped": cropped,
        "annotated_rgb": annotated,
        "circle_found": circle_found,
        "circle_info": circle_info,
        "white_plate": white_plate,
        "white_ratio": white_ratio,
    }

# INFERENCE
@torch.no_grad()
def predict(model: nn.Module, tensor: torch.Tensor,
            classes: list[str]) -> tuple[str, float, np.ndarray]:
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], float(probs[idx]) * 100, probs.cpu().numpy()

# UI HELPERS
LABEL_COLORS = {
    "crack": (220, 50, 50),
    "spot": (220, 180, 30),
    "pinhole": (50, 120, 220),
    "normal": (30, 200, 80),
}

def apply_threshold(label: str, conf: float,
                    threshold: float = DEFAULT_CONF_THRESH) -> tuple[str, str | None]:
    if label in DEFECT_CLASSES_FOR_THRESHOLD and conf < threshold:
        return "normal", label
    return label, None

def show_result(label: str, conf: float, probs: np.ndarray,
                classes: list[str], circle_found: bool,
                original_label: str | None = None,
                threshold: float = DEFAULT_CONF_THRESH) -> None:
    if not circle_found:
        st.error("Lingkaran piring tidak terdeteksi. Kemungkinan: piring tidak berbentuk lingkaran, terlalu rusak, atau bukan piring.")
        st.info("Model tetap dijalankan pada gambar penuh. Hasil mungkin kurang akurat.")
    
    if label == "normal" and original_label:
        st.markdown("### Prediksi: `NORMAL (TIDAK ADA DEFEK)`")
        st.info(
            f"Model mendeteksi indikasi '{original_label}' dengan confidence {conf:.1f}%, "
            f"namun di bawah threshold ({threshold:.0f}%) — piring dinyatakan NORMAL."
        )
    else:
        color_hex = "#{:02x}{:02x}{:02x}".format(*LABEL_COLORS.get(label, (150, 150, 150)))
        st.markdown(
            f"### Prediksi: "
            f"<span style='color:{color_hex};font-weight:700'>{label.upper()}</span> "
            f"— {conf:.1f}%",
            unsafe_allow_html=True,
        )

    # Progress bar per kelas
    for i, cls in enumerate(classes):
        pct = float(probs[i]) * 100
        st.progress(float(probs[i]), text=f"{cls}: {pct:.1f}%")

def show_pipeline_images(orig_rgb: np.ndarray, result: dict) -> None:
    st.markdown("#### Pipeline Pemrosesan Gambar")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(result["annotated_rgb"], caption="1. Original + Deteksi Lingkaran", use_container_width=True)
    with col2:
        st.image(result["gray"], caption="2. Grayscale", use_container_width=True, clamp=True)
    with col3:
        st.image(result["inverted"], caption="3. Inversi Warna", use_container_width=True, clamp=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(result["exposed"], caption="4. Exposure (adjustable)", use_container_width=True, clamp=True)
    with col5:
        st.image(result["sharpened"], caption="5. Sharpened (adjustable)", use_container_width=True, clamp=True)
    with col6:
        lbl = "6. Crop — Lingkaran Terdeteksi" if result["circle_found"] else "6. Crop Gagal"
        st.image(result["cropped"], caption=lbl, use_container_width=True, clamp=True)

# STREAMLIT APP
def main() -> None:
    st.set_page_config(
        page_title="Plate Sight - Deteksi Cacat Piring Keramik",
        page_icon="img/logo.png",
        layout="wide",
    )
    
    # Inject custom CSS for better font and clean layout (no gradients, no bright effects)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global font settings */
    .stApp, .stApp > header, .stApp > .main .block-container, p, h1, h2, h3, h4, h5, h6, label, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Clean headings */
    h1, h2, h3, h4 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        border: 1px solid #d1d5db !important;
    }
    .stButton>button:hover {
        border-color: #9ca3af !important;
    }
    
    /* Remove default bright focus rings if any, keep it subtle */
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
        box-shadow: 0 0 0 1px #9ca3af !important;
    }
    
    /* Kurangi jarak antar kolom header logo-title */
    div[data-testid="stHorizontalBlock"]:has(img) {
        gap: 0.5rem !important;
        align-items: center !important;
    }
    /* Clean up tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with Logo
    col_logo, col_title = st.columns([1, 7])
    with col_logo:
        try:
            st.image("img/logo.png", width=110)
        except Exception:
            pass # Fallback if logo is missing
    with col_title:
        st.title("Plate Sight")
        st.caption("Sistem deteksi cacat pada piring keramik berwarna putih menggunakan model DINOv2.")
        # Disclaimer: jelas dan singkat mengenai keterbatasan aplikasi
        st.info(
            "Disclaimer: Aplikasi ini hanya mendeteksi defek berupa retak atau noda pada piring putih tanpa pola. "
            "Hasil mungkin tidak akurat untuk piring berpola, berwarna lain, atau material selain keramik."
        )

    init_db()
    model, classes = load_model()

    tab_upload, tab_webcam, tab_history, tab_analytics = st.tabs([
        "Upload Gambar",
        "Webcam Live",
        "Riwayat Prediksi",
        "Analitik",
    ])

    # TAB 1 — UPLOAD
    with tab_upload:
        st.subheader("Upload Gambar Keramik")

        with st.expander("Pengaturan Pipeline", expanded=False):
            uc1, uc2, uc3 = st.columns(3)
            with uc1:
                up_exposure = st.slider(
                    "Exposure (alpha)", min_value=0.0, max_value=3.0,
                    value=DEFAULT_EXPOSURE, step=0.05, key="up_exposure",
                    help="Kecerahan setelah inversi. >1 = terang, <1 = gelap."
                )
            with uc2:
                up_sharpen = st.slider(
                    "Sharpening", min_value=1.0, max_value=5.0,
                    value=DEFAULT_SHARPEN_AMT, step=0.1, key="up_sharpen",
                    help="Intensitas Unsharp Mask."
                )
            with uc3:
                up_threshold = st.slider(
                    "Threshold Defek (%)", min_value=30.0, max_value=99.0,
                    value=DEFAULT_CONF_THRESH, step=1.0, key="up_threshold",
                    help="Confidence di bawah ini dianggap NORMAL."
                )
            st.caption(
                f"Aktif: Exposure x{up_exposure:.2f} | Sharpen x{up_sharpen:.1f} | Threshold {up_threshold:.0f}%"
            )

        uploaded = st.file_uploader(
            "Pilih gambar (.jpg / .jpeg / .png)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            img_array = np.array(pil_img)

            with st.spinner("Memproses gambar..."):
                result = preprocess(img_array, exposure=up_exposure, sharpen_amt=up_sharpen)

                if result["circle_found"]:
                    if not result["white_plate"]:
                        st.warning(
                            "Area lingkaran terdeteksi, tetapi tidak terlihat sebagai piring putih. "
                            "Pastikan hanya mengunggah gambar piring putih tanpa pola berwarna.")
                        st.caption(f"Persentase area putih di lingkaran: {result['white_ratio']:.1f}%")
                        st.divider()
                        show_pipeline_images(img_array, result)
                    else:
                        raw_label, conf, probs = predict(model, result["tensor"], classes)
                        final_label, original_label = apply_threshold(raw_label, conf, up_threshold)

                        st.divider()
                        # Tampilkan hasil prediksi terlebih dahulu, lalu pipeline pemrosesan
                        show_result(final_label, conf, probs, classes,
                                    result["circle_found"], original_label, up_threshold)
                        st.divider()
                        show_pipeline_images(img_array, result)

                        save_prediction(
                            "upload", uploaded.name, final_label, conf,
                            result["circle_found"], up_exposure, up_sharpen,
                            encode_image_bytes(img_array),
                        )
                        st.success("Prediksi berhasil disimpan ke database.")
                else:
                    st.warning("Lingkaran piring tidak terdeteksi — model tidak dijalankan dan prediksi tidak tersedia.")
                    st.divider()
                    show_pipeline_images(img_array, result)

    # TAB 2 — WEBCAM LIVE
    with tab_webcam:
        st.subheader("Live Webcam Detection")

        if not WEBRTC_AVAILABLE:
            st.error(
                "Paket `streamlit-webrtc` dan `av` belum terinstall.\n\n"
                "Jalankan: `pip install streamlit-webrtc av`"
            )
        else:
            with st.expander("Pengaturan Pipeline", expanded=True):
                wc1, wc2, wc3 = st.columns(3)
                with wc1:
                    wc_exposure = st.slider(
                        "Exposure (alpha)", min_value=0.0, max_value=3.0,
                        value=DEFAULT_EXPOSURE, step=0.05, key="wc_exposure",
                        help="Kecerahan setelah inversi. >1 = terang, <1 = gelap."
                    )
                with wc2:
                    wc_sharpen = st.slider(
                        "Sharpening", min_value=1.0, max_value=5.0,
                        value=DEFAULT_SHARPEN_AMT, step=0.1, key="wc_sharpen",
                        help="Intensitas Unsharp Mask."
                    )
                with wc3:
                    wc_threshold = st.slider(
                        "Threshold Defek (%)", min_value=30.0, max_value=99.0,
                        value=DEFAULT_CONF_THRESH, step=1.0, key="wc_threshold",
                        help="Confidence di bawah ini dianggap NORMAL."
                    )
                st.caption(
                    f"Pipeline aktif: BGR -> Grayscale -> Inversi -> Exposure x{wc_exposure:.2f} -> Sharpen x{wc_sharpen:.1f} -> Crop -> Model -> Threshold {wc_threshold:.0f}%"
                )
                st.divider()
                st.info(
                    "Setelah mengubah slider, matikan kamera lalu nyalakan kembali "
                    "supaya nilai baru langsung diterapkan ke kamera."
                )

            exposure = wc_exposure
            sharpen_amt = wc_sharpen
            threshold = wc_threshold

            for key, default in [
                ("webcam_capture", None),
                ("webcam_session_counts", {}),
                ("webcam_session_total", 0),
                ("webcam_last_save_time", 0.0),
                ("webcam_fps_deque", deque(maxlen=30)),
                ("webcam_auto_save", False),
                ("webcam_save_cooldown", AUTOSAVE_COOLDOWN),
            ]:
                if key not in st.session_state:
                    st.session_state[key] = default

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                auto_save = st.toggle("Auto-Save ke DB", value=st.session_state.webcam_auto_save)
                st.session_state.webcam_auto_save = auto_save
            with col_ctrl2:
                save_cd = st.number_input(
                    "Cooldown Auto-Save (detik)", min_value=1.0, max_value=60.0,
                    value=st.session_state.webcam_save_cooldown, step=1.0,
                )
                st.session_state.webcam_save_cooldown = save_cd
            with col_ctrl3:
                show_overlay = st.checkbox("Tampilkan Overlay Lengkap", value=True)

            ph_metrics = st.empty()
            ph_status = st.empty()

            class DefectProcessor(VideoProcessorBase):
                def __init__(self) -> None:
                    self._model = model
                    self._classes = classes
                    self._lock = threading.Lock()
                    self.result = {"label": "-", "conf": 0.0, "circle": False,
                                   "original_label": None, "probs": None}
                    self.last_frame_rgb = None
                    self._frame_times: deque = deque(maxlen=30)
                    self._last_save = 0.0
                    self._last_proc = 0.0

                @property
                def _exposure(self):
                    return exposure

                @property
                def _sharpen(self):
                    return sharpen_amt

                @property
                def _threshold(self):
                    return threshold

                def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                    t0 = time.perf_counter()
                    img_bgr = frame.to_ndarray(format="bgr24")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    with self._lock:
                        self.last_frame_rgb = img_rgb.copy()

                    if (t0 - self._last_proc) < _MIN_FRAME_INTERVAL:
                        return frame
                    self._last_proc = t0

                    res = preprocess(img_rgb, exposure=self._exposure, sharpen_amt=self._sharpen)

                    if res.get("circle_found") and res.get("white_plate"):
                        with torch.no_grad():
                            raw_label, conf, probs = predict(self._model, res["tensor"], self._classes)
                        final_label, original_label = apply_threshold(raw_label, conf, self._threshold)
                    else:
                        raw_label = "-"
                        conf = 0.0
                        probs = np.zeros(len(self._classes), dtype=np.float32)
                        final_label = "-"
                        original_label = None

                    is_live_defect = res.get("circle_found") and res.get("white_plate") and final_label in ("crack", "spot")
                    with self._lock:
                        self.result = {
                            "label": final_label,
                            "conf": conf,
                            "circle": res["circle_found"],
                            "original_label": original_label,
                            "probs": probs,
                            "white_plate": res.get("white_plate", False),
                            "white_ratio": res.get("white_ratio", 0.0),
                            "live_defect": 1 if is_live_defect else 0,
                        }

                    try:
                        global LIVE_DEFECT
                        LIVE_DEFECT = 1 if is_live_defect else 0
                    except Exception:
                        pass

                    now = time.time()
                    auto_save_on = getattr(self, "_auto_save", False)
                    cooldown = getattr(self, "_save_cooldown", AUTOSAVE_COOLDOWN)
                    # Hanya auto-save ketika lingkaran piring terdeteksi
                    if auto_save_on and res.get("circle_found") and (now - self._last_save) >= cooldown:
                        save_prediction("webcam_live", "live_frame", final_label, conf,
                                        res["circle_found"], self._exposure, self._sharpen,
                                        encode_image_bytes(img_rgb, quality=65))
                        self._last_save = now

                    self._frame_times.append(time.perf_counter())

                    if res["circle_found"]:
                        out_bgr = cv2.cvtColor(res["cropped"], cv2.COLOR_GRAY2BGR)
                    else:
                        out_bgr = img_bgr.copy()

                    if show_overlay:
                        h0, w0 = img_bgr.shape[:2]
                        h1, w1 = out_bgr.shape[:2]

                        if res["circle_found"] and res["circle_info"]:
                            cx, cy, r = res["circle_info"]
                            sx, sy = w1 / w0, h1 / h0
                            cx2 = int(cx * sx); cy2 = int(cy * sy); r2 = int(r * min(sx, sy))
                            c_color = (0, 200, 80) if final_label == "normal" else (0, 60, 220)
                            cv2.circle(out_bgr, (cx2, cy2), r2, c_color, 2)
                            cv2.circle(out_bgr, (cx2, cy2), 5, c_color, -1)

                        if not res["circle_found"]:
                            text = "NO CIRCLE DETECTED"
                            text_color = (0, 165, 255)
                        elif final_label == "normal":
                            text = f"NORMAL ({raw_label} {conf:.0f}%)"
                            text_color = (30, 220, 80)
                        else:
                            text = f"{final_label.upper()} {conf:.0f}%"
                            text_color = (0, 60, 220)

                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                        cv2.rectangle(out_bgr, (8, 8), (tw + 16, th + 20), (0, 0, 0), -1)
                        cv2.putText(out_bgr, text, (12, th + 12),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.9, text_color, 2, cv2.LINE_AA)

                        info_txt = f"EXP x{self._exposure:.2f} SHP x{self._sharpen:.1f} THR {self._threshold:.0f}%"
                        cv2.putText(out_bgr, info_txt, (10, h1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                    if len(self._frame_times) >= 2:
                        elapsed = self._frame_times[-1] - self._frame_times[0]
                        fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0
                        cv2.putText(out_bgr, f"FPS {fps:.1f}",
                                    (out_bgr.shape[1] - 90, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)

                    t1 = time.perf_counter()
                    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

                def get_fps(self) -> float:
                    ft = self._frame_times
                    if len(ft) < 2:
                        return 0.0
                    elapsed = ft[-1] - ft[0]
                    return (len(ft) - 1) / elapsed if elapsed > 0 else 0.0

            _METERED_USERNAME = os.environ.get("METERED_USERNAME", "")
            _METERED_PASSWORD = os.environ.get("METERED_PASSWORD", "")
            # Build ICE servers list; only include TURN entries when credentials present
            ice_servers = [
                {"urls": ["stun:stun.relay.metered.ca:80"]},
            ]

            if _METERED_USERNAME and _METERED_PASSWORD:
                ice_servers.extend([
                    {
                        "urls": ["turn:standard.relay.metered.ca:80"],
                        "username": _METERED_USERNAME,
                        "credential": _METERED_PASSWORD,
                    },
                    {
                        "urls": ["turn:standard.relay.metered.ca:80?transport=tcp"],
                        "username": _METERED_USERNAME,
                        "credential": _METERED_PASSWORD,
                    },
                    {
                        "urls": ["turn:standard.relay.metered.ca:443"],
                        "username": _METERED_USERNAME,
                        "credential": _METERED_PASSWORD,
                    },
                    {
                        "urls": ["turns:standard.relay.metered.ca:443?transport=tcp"],
                        "username": _METERED_USERNAME,
                        "credential": _METERED_PASSWORD,
                    },
                ])
            else:
                # Jika kredensial TURN kosong, abaikan TURN servers (cukup STUN)
                st.info("TURN credentials tidak ditemukan di .env — TURN servers diabaikan.")

            RTC_CONFIG = RTCConfiguration({"iceServers": ice_servers})

            ctx = webrtc_streamer(
                key="ceramic-defect-live",
                video_processor_factory=DefectProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15},
                    },
                    "audio": False,
                },
                async_processing=True,
            )

            if ctx.video_processor:
                ctx.video_processor._auto_save = auto_save
                ctx.video_processor._save_cooldown = save_cd

                res = ctx.video_processor.result
                fps = ctx.video_processor.get_fps()

                lbl = res["label"]
                if lbl != "-":
                    st.session_state.webcam_session_counts[lbl] = \
                        st.session_state.webcam_session_counts.get(lbl, 0) + 1
                    st.session_state.webcam_session_total += 1

                with ph_metrics.container():
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("FPS", f"{fps:.1f}")
                    mc2.metric("Label", res["label"].upper() if res["label"] != "-" else "—")
                    mc3.metric("Confidence", f"{res['conf']:.1f}%" if res["conf"] > 0 else "—")
                    mc4.metric("Lingkaran", "Terdeteksi" if res["circle"] else "Tidak")
                    mc5.metric("Piring Putih", "Ya" if res.get("white_plate") else "Tidak")

                with ph_status.container():
                    if not res["circle"]:
                        st.error("Lingkaran piring tidak terdeteksi — arahkan kamera ke piring keramik.")
                    elif res["circle"] and not res.get("white_plate"):
                        st.warning(
                            "Area lingkaran terdeteksi, tetapi tidak terlihat seperti piring putih. "
                            "Arahkan kamera ke piring putih tanpa corak atau warna lain."
                        )
                    elif res["label"] == "normal":
                        st.success(
                            f"NORMAL — Deteksi '{res['original_label']}' "
                            f"({res['conf']:.1f}%) di bawah threshold {threshold:.0f}%"
                        )
                    elif res["label"] != "-":
                        st.error(f"DEFEK: {res['label'].upper()} — Confidence {res['conf']:.1f}%")

                    if st.session_state.webcam_session_total > 0:
                        parts = [f"`{k}`: {v}" for k, v in st.session_state.webcam_session_counts.items()]
                        st.caption(f"Session frames: {st.session_state.webcam_session_total} | " + " | ".join(parts))

                st.divider()
                col_cap, col_save, col_reset = st.columns(3)
                with col_cap:
                    if st.button("Capture & Analisis Frame"):
                        with ctx.video_processor._lock:
                            frm = ctx.video_processor.last_frame_rgb
                        if frm is not None:
                            st.session_state.webcam_capture = frm.copy()
                        else:
                            st.warning("Belum ada frame. Tunggu beberapa detik.")
                with col_save:
                    if st.button("Simpan Prediksi Sekarang"):
                        with ctx.video_processor._lock:
                            frm_save = ctx.video_processor.last_frame_rgb
                        img_bytes_now = encode_image_bytes(frm_save, quality=65) if frm_save is not None else None
                        if res["circle"]:
                            save_prediction("webcam", "live_frame",
                                            res["label"], res["conf"], res["circle"],
                                            exposure, sharpen_amt, img_bytes_now)
                            st.success(f"Disimpan: {res['label']} ({res['conf']:.1f}%)")
                        else:
                            st.warning("Lingkaran piring tidak terdeteksi — prediksi tidak disimpan ke database.")
                with col_reset:
                    if st.button("Reset Session Stats"):
                        st.session_state.webcam_session_counts = {}
                        st.session_state.webcam_session_total = 0
                        st.rerun()

                if auto_save:
                    st.info(f"Auto-save aktif setiap {save_cd:.0f} detik (hanya prediksi valid).")

            if st.session_state.webcam_capture is not None:
                st.divider()
                st.markdown("### Analisis Frame Terakhir")

                cap_img = st.session_state.webcam_capture
                with st.spinner("Memproses frame..."):
                    cap_result = preprocess(cap_img, exposure=exposure, sharpen_amt=sharpen_amt)
                    if cap_result["circle_found"]:
                        if not cap_result.get("white_plate"):
                            st.warning(
                                "Area lingkaran terdeteksi, tetapi tidak terlihat seperti piring putih. "
                                "Prediksi tidak dijalankan untuk gambar non-putih."
                            )
                            st.caption(f"Persentase area putih di lingkaran: {cap_result['white_ratio']:.1f}%")
                            st.divider()
                            show_pipeline_images(cap_img, cap_result)
                        else:
                            cap_raw, cap_conf, cap_probs = predict(model, cap_result["tensor"], classes)
                            cap_final, cap_orig = apply_threshold(cap_raw, cap_conf, threshold)

                            # Tampilkan hasil prediksi terlebih dahulu, lalu pipeline pemrosesan
                            show_result(cap_final, cap_conf, cap_probs, classes,
                                        cap_result["circle_found"], cap_orig, threshold)
                            st.divider()
                            show_pipeline_images(cap_img, cap_result)
                    else:
                        st.warning("Lingkaran piring tidak terdeteksi — model tidak dijalankan dan prediksi tidak tersedia.")
                        st.divider()
                        show_pipeline_images(cap_img, cap_result)

                col_sv, col_cl = st.columns(2)
                with col_sv:
                    if st.button("Simpan Hasil Capture"):
                        if cap_result["circle_found"]:
                            save_prediction("webcam_capture", "captured_frame",
                                            cap_final, cap_conf, cap_result["circle_found"],
                                            exposure, sharpen_amt,
                                            encode_image_bytes(cap_img, quality=75))
                            st.success("Prediksi capture disimpan.")
                        else:
                            st.warning("Lingkaran piring tidak terdeteksi — prediksi capture tidak disimpan.")
                with col_cl:
                    if st.button("Hapus Capture"):
                        st.session_state.webcam_capture = None
                        st.rerun()

    # TAB 3 — HISTORY
    with tab_history:
        st.subheader("Riwayat Prediksi (50 terakhir)")

        col_r, col_c, col_e = st.columns(3)
        with col_r:
            if st.button("Refresh"):
                st.rerun()
        with col_c:
            if st.button("Hapus Semua"):
                clear_history()
                st.success("Riwayat dihapus.")
                st.rerun()
        with col_e:
            if st.button("Export All Images to scannedplate/"):
                rows_all = get_history(limit=100000)
                exported = 0
                SCANNED_DIR.mkdir(parents=True, exist_ok=True)
                for r in rows_all:
                    pred_id = r[0]
                    label = r[3]
                    created = r[8] or datetime.now().isoformat()
                    img_bytes = r[9]
                    if img_bytes:
                        try:
                            ts = created.replace(":", "").replace("-", "").replace("T", "_")
                            fname = f"pred_{pred_id}_{label}_{ts}.jpg"
                            out_path = SCANNED_DIR / fname
                            with open(out_path, "wb") as f:
                                f.write(img_bytes)
                            exported += 1
                        except Exception:
                            pass
                st.success(f"Ekspor selesai — {exported} gambar disimpan ke {SCANNED_DIR}/")
            if st.button("Download All as ZIP"):
                rows_all = get_history(limit=100000)
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for r in rows_all:
                        pred_id = r[0]
                        label = r[3]
                        created = r[8] or datetime.now().isoformat()
                        img_bytes = r[9]
                        if img_bytes:
                            try:
                                ts = created.replace(":", "").replace("-", "").replace("T", "_")
                                fname = f"pred_{pred_id}_{label}_{ts}.jpg"
                                zf.writestr(fname, img_bytes)
                            except Exception:
                                pass
                zip_buf.seek(0)
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buf.getvalue(),
                    file_name="scannedplate_all.zip",
                    mime="application/zip",
                )

        rows = get_history()
        if rows:
            all_labels = sorted({r[3] for r in rows})
            filter_label = st.multiselect("Filter Label", options=all_labels, default=all_labels)
            filtered = [r for r in rows if r[3] in filter_label]

            # View mode toggle
            view_mode = st.radio(
                "Tampilan", ["Kartu dengan Gambar", "Tabel"],
                horizontal=True, label_visibility="collapsed"
            )

            st.divider()

            if view_mode == "Tabel":
                # Legacy table view (tanpa gambar)
                st.table([
                    {
                        "ID": r[0],
                        "Source": r[1],
                        "File": r[2],
                        "Label": r[3],
                        "Conf (%)": f"{r[4]:.2f}",
                        "Circle": "Ya" if r[5] else "Tidak",
                        "Exposure": f"x{r[6]:.2f}" if r[6] else "-",
                        "Sharpen": f"x{r[7]:.1f}" if r[7] else "-",
                        "Waktu": r[8],
                        "Gambar": "✅" if r[9] else "❌",
                    }
                    for r in filtered
                ])
            else:
                # Card view with thumbnail + full image modal
                # Inject CSS for card styling
                st.markdown("""
                <style>
                .hist-card {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 10px;
                    padding: 12px 16px;
                    margin-bottom: 10px;
                }
                .hist-label-crack  { color: #f87171; font-weight: 700; }
                .hist-label-spot   { color: #fbbf24; font-weight: 700; }
                .hist-label-pinhole{ color: #60a5fa; font-weight: 700; }
                .hist-label-normal { color: #34d399; font-weight: 700; }
                .hist-meta { font-size: 0.82em; color: #9ca3af; margin-top: 4px; }
                </style>
                """, unsafe_allow_html=True)

                LABEL_CSS = {
                    "crack": "hist-label-crack",
                    "spot": "hist-label-spot",
                    "pinhole": "hist-label-pinhole",
                    "normal": "hist-label-normal",
                }

                for r in filtered:
                    # r: id[0] source[1] filename[2] label[3] conf[4]
                    #    circle_found[5] exposure[6] sharpen[7] created_at[8] image_data[9]
                    pred_id   = r[0]
                    source    = r[1]
                    fname     = r[2]
                    label     = r[3]
                    conf      = r[4]
                    circle    = bool(r[5])
                    exposure  = r[6]
                    sharpen   = r[7]
                    created   = r[8]
                    img_bytes = r[9]

                    css_cls = LABEL_CSS.get(label, "hist-label-normal")
                    conf_bar = min(conf / 100.0, 1.0)

                    col_thumb, col_info = st.columns([1, 3])

                    with col_thumb:
                        if img_bytes:
                            try:
                                pil_thumb = Image.open(io.BytesIO(img_bytes))
                                st.image(pil_thumb, use_container_width=True)
                            except Exception:
                                st.caption("⚠️ Error load gambar")
                        else:
                            # Placeholder jika tidak ada gambar
                            st.markdown(
                                "<div style='background:rgba(255,255,255,0.05);border-radius:6px;"
                                "aspect-ratio:1;display:flex;align-items:center;justify-content:center;"
                                "font-size:1.8em;'>📷</div>",
                                unsafe_allow_html=True
                            )

                    with col_info:
                        st.markdown(
                            f"**#{pred_id}** &nbsp; "
                            f"<span class='{css_cls}'>{label.upper()}</span> &nbsp; "
                            f"— &nbsp; **{conf:.1f}%**",
                            unsafe_allow_html=True,
                        )
                        st.progress(conf_bar)
                        st.markdown(
                            f"<div class='hist-meta'>"
                            f"🕐 {created} &nbsp;|&nbsp; "
                            f"📂 {source} &nbsp;|&nbsp; "
                            f"🗂️ {fname} &nbsp;|&nbsp; "
                            f"⭕ {'Terdeteksi' if circle else 'Tidak'} &nbsp;|&nbsp; "
                            f"EXP x{exposure:.2f} &nbsp;|&nbsp; SHP x{sharpen:.1f}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        if img_bytes:
                            st.download_button(
                                label="⬇️ Download Gambar",
                                data=img_bytes,
                                file_name=f"pred_{pred_id}_{label}_{created[:10]}.jpg",
                                mime="image/jpeg",
                                key=f"dl_{pred_id}",
                            )

                    st.divider()
        else:
            st.info("Belum ada prediksi tersimpan.")

    # TAB 4 — ANALITIK
    with tab_analytics:
        st.subheader("Analitik Prediksi")

        if st.button("Refresh Analitik"):
            st.rerun()

        stats = get_stats()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Prediksi", stats["total"])
        m2.metric("Rata-rata Confidence", f"{stats['avg_conf']:.1f}%")
        m3.metric("Jumlah Kelas Terdeteksi", len(stats["by_label"]))

        if stats["by_label"]:
            st.divider()
            st.markdown("#### Distribusi Label")

            label_data = {row[0]: row[1] for row in stats["by_label"]}
            st.bar_chart(label_data)

            st.markdown("#### Detail per Label")
            total = stats["total"] or 1
            for lbl, cnt in stats["by_label"]:
                pct = cnt / total * 100
                st.progress(pct / 100, text=f"{lbl}: {cnt} prediksi ({pct:.1f}%)")
        else:
            st.info("Belum ada data untuk dianalisis. Jalankan beberapa prediksi terlebih dahulu.")

        rows = get_history(limit=200)
        if rows:
            st.divider()
            st.markdown("#### Statistik Confidence per Label")
            from collections import defaultdict
            label_confs: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                label_confs[r[3]].append(r[4])

            stat_rows = []
            for lbl, confs in sorted(label_confs.items()):
                stat_rows.append({
                    "Label": lbl,
                    "Count": len(confs),
                    "Min %": f"{min(confs):.1f}",
                    "Max %": f"{max(confs):.1f}",
                    "Avg %": f"{sum(confs)/len(confs):.1f}",
                })
            st.table(stat_rows)

if __name__ == "__main__":
    main()