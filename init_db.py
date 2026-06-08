"""
Ceramic Tile Defect Detector — v2
Pipeline: Image/Webcam → Grayscale → (Detect Circle) → Invert → Exposure (adjustable) → Sharpen (adjustable) → Crop (Apply Mask) → DINOv2 → Prediction → Threshold (adjustable) → SQLite
"""

import os
import sqlite3
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

# ── streamlit-webrtc (live webcam) ────────────────────────────────────────────
try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = Path("best_model_dinov2.pth")
IMG_SIZE   = 224
DB_PATH    = "predictions.db"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES_2 = ["crack", "spot"]
CLASSES_3 = ["crack", "pinhole", "spot"]

# Default pipeline params
DEFAULT_EXPOSURE      = 0.5   # alpha untuk convertScaleAbs
DEFAULT_SHARPEN_AMT   = 3.0   # weight untuk addWeighted
DEFAULT_CONF_THRESH   = 95.0  # %

DEFECT_CLASSES_FOR_THRESHOLD = ["crack", "spot", "pinhole"]

# Auto-save cooldown (detik) agar DB tidak membludak saat live
AUTOSAVE_COOLDOWN = 5.0

# Max FPS untuk VideoProcessor (batasi beban CPU/GPU)
MAX_FPS = 3
_MIN_FRAME_INTERVAL = 1.0 / MAX_FPS  # ~0.333 s

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
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
            created_at   TEXT    NOT NULL
        )
    """)
    # Migrasi: tambah kolom baru jika belum ada (backward compat)
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN exposure REAL DEFAULT 1.3")
        conn.execute("ALTER TABLE predictions ADD COLUMN sharpen  REAL DEFAULT 1.5")
    except Exception:
        pass
    conn.commit()
    conn.close()


def save_prediction(source: str, filename: str, label: str,
                    confidence: float, circle_found: bool = False,
                    exposure: float = 1.3, sharpen: float = 1.5) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions "
        "(source, filename, label, confidence, circle_found, exposure, sharpen, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (source, filename or "-", label, round(confidence, 4),
         int(circle_found), round(exposure, 2), round(sharpen, 2),
         datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 50) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, source, filename, label, confidence, circle_found, exposure, sharpen, created_at "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_stats() -> dict:
    """Ambil statistik agregat dari database."""
    conn = sqlite3.connect(DB_PATH)
    total   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    by_label = conn.execute(
        "SELECT label, COUNT(*) FROM predictions GROUP BY label ORDER BY COUNT(*) DESC"
    ).fetchall()
    avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    conn.close()
    return {"total": total, "by_label": by_label, "avg_conf": avg_conf or 0.0}


def clear_history() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — auto-detect num_classes dari checkpoint
# ══════════════════════════════════════════════════════════════════════════════
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


@st.cache_resource(show_spinner="⏳ Memuat model DINOv2…")
def load_model() -> tuple[nn.Module, list[str]]:
    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    classes     = _get_classes_from_checkpoint(MODEL_PATH)
    num_classes = len(classes)
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        img_size=IMG_SIZE,
    )
    in_features = getattr(model, "num_features", None) or getattr(model, "embed_dim", None)
    model.head  = nn.Linear(in_features, num_classes)
    state_dict  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model, classes


# ══════════════════════════════════════════════════════════════════════════════
# CIRCLE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_circle_mask(gray: np.ndarray) -> tuple[np.ndarray | None, bool, tuple | None]:
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp        = 1.2,
        minDist   = min(h, w) * 0.4,
        param1    = 100,
        param2    = 40,
        minRadius = int(min(h, w) * 0.2),
        maxRadius = int(min(h, w) * 0.55),
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


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE (exposure & sharpen bisa dikontrol)
# ══════════════════════════════════════════════════════════════════════════════
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(img_rgb: np.ndarray,
               exposure: float = DEFAULT_EXPOSURE,
               sharpen_amt: float = DEFAULT_SHARPEN_AMT) -> dict:
    """
    Pipeline lengkap dengan exposure & sharpen yang dapat dikonfigurasi.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Deteksi lingkaran
    mask, circle_found, circle_info = detect_circle_mask(gray)

    # 3. Inversi Warna
    inverted = cv2.bitwise_not(gray)

    # 4. Exposure (alpha = exposure factor)
    exposed = cv2.convertScaleAbs(inverted, alpha=exposure, beta=0)

    # 5. Sharpen via Unsharp Mask
    blurred   = cv2.GaussianBlur(exposed, (0, 0), sigmaX=3)
    # sharpen_amt: bobot gambar tajam, (sharpen_amt - 1) dikurangi dari blurred
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

    return {
        "tensor"       : tensor,
        "gray"         : gray,
        "inverted"     : inverted,
        "exposed"      : exposed,
        "sharpened"    : sharpened,
        "cropped"      : cropped,
        "annotated_rgb": annotated,
        "circle_found" : circle_found,
        "circle_info"  : circle_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model: nn.Module, tensor: torch.Tensor,
            classes: list[str]) -> tuple[str, float, np.ndarray]:
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    idx    = probs.argmax().item()
    return classes[idx], float(probs[idx]) * 100, probs.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
LABEL_COLORS = {
    "crack"  : (220, 50,  50),
    "spot"   : (220, 180, 30),
    "pinhole": (50,  120, 220),
    "normal" : (30,  200, 80),
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
        st.markdown(f"### Prediksi: `NORMAL (TIDAK ADA DEFEK)`")
        st.info(
            f"Model mendeteksi indikasi '{original_label}' dengan confidence {conf:.1f}%, "
            f"namun di bawah threshold ({threshold:.0f}%) — piring dinyatakan NORMAL."
        )
    else:
        color_hex = "#{:02x}{:02x}{:02x}".format(*LABEL_COLORS.get(label, (150, 150, 150)))
        st.markdown(
            f"### Prediksi: "
            f"<span style='color:{color_hex};font-weight:700'>{label.upper()}</span>"
            f" — {conf:.1f}%",
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
        st.image(result["annotated_rgb"],
                 caption="1. Original + Deteksi Lingkaran", use_container_width=True)
    with col2:
        st.image(result["gray"],
                 caption="2. Grayscale", use_container_width=True, clamp=True)
    with col3:
        st.image(result["inverted"],
                 caption="3. Inversi Warna", use_container_width=True, clamp=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(result["exposed"],
                 caption="4. Exposure (adjustable)", use_container_width=True, clamp=True)
    with col5:
        st.image(result["sharpened"],
                 caption="5. Sharpened (adjustable)", use_container_width=True, clamp=True)
    with col6:
        lbl = ("6. Crop — Lingkaran Terdeteksi"
               if result["circle_found"] else "6. Crop Gagal")
        st.image(result["cropped"], caption=lbl, use_container_width=True, clamp=True)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(
        page_title="Ceramic Defect Detector",
        page_icon="",
        layout="centered",
    )

    st.title("Ceramic Tile Defect Detector")

    init_db()
    model, classes = load_model()

    st.caption(
        f"Model: DINOv2 ViT-Small · Device: {DEVICE} · "
        f"Kelas ({len(classes)}): {', '.join(classes)}"
    )

    tab_upload, tab_webcam, tab_history, tab_analytics = st.tabs([
        "Upload Gambar",
        "Webcam Live",
        "Riwayat Prediksi",
        "Analitik",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.subheader("Upload Gambar Keramik")

        # ── Pipeline Controls (inline) ─────────────────────────────────────────
        with st.expander("Pipeline Controls", expanded=False):
            uc1, uc2, uc3 = st.columns(3)
            with uc1:
                up_exposure = st.slider(
                    "Exposure (alpha)", min_value=0.5, max_value=3.0,
                    value=DEFAULT_EXPOSURE, step=0.05, key="up_exposure",
                    help="Kecerahan setelah inversi. >1 = terang, <1 = gelap.",
                )
            with uc2:
                up_sharpen = st.slider(
                    "Sharpening", min_value=1.0, max_value=5.0,
                    value=DEFAULT_SHARPEN_AMT, step=0.1, key="up_sharpen",
                    help="Intensitas Unsharp Mask.",
                )
            with uc3:
                up_threshold = st.slider(
                    "Threshold Defek (%)", min_value=30.0, max_value=99.0,
                    value=DEFAULT_CONF_THRESH, step=1.0, key="up_threshold",
                    help="Confidence di bawah ini → dianggap NORMAL.",
                )
            st.caption(
                f"Aktif: Exposure **×{up_exposure:.2f}** · "
                f"Sharpen **×{up_sharpen:.1f}** · "
                f"Threshold **{up_threshold:.0f}%**"
            )

        uploaded = st.file_uploader(
            "Pilih gambar (.jpg / .jpeg / .png)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is not None:
            pil_img   = Image.open(uploaded).convert("RGB")
            img_array = np.array(pil_img)

            with st.spinner("Memproses gambar…"):
                result             = preprocess(img_array, exposure=up_exposure, sharpen_amt=up_sharpen)
                raw_label, conf, probs = predict(model, result["tensor"], classes)
                final_label, original_label = apply_threshold(raw_label, conf, up_threshold)

            st.divider()
            show_pipeline_images(img_array, result)
            st.divider()
            show_result(final_label, conf, probs, classes,
                        result["circle_found"], original_label, up_threshold)

            save_prediction(
                "upload", uploaded.name, final_label, conf,
                result["circle_found"], up_exposure, up_sharpen
            )
            st.success("Prediksi disimpan ke database.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — WEBCAM LIVE (UPGRADED)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_webcam:
        st.subheader("Live Webcam Detection")

        if not WEBRTC_AVAILABLE:
            st.error(
                "Paket `streamlit-webrtc` dan `av` belum terinstall.\n\n"
                "Jalankan: `pip install streamlit-webrtc av`"
            )
        else:
            # ── Pipeline Controls (inline) ─────────────────────────────────────
            with st.expander("Pipeline Controls", expanded=True):
                wc1, wc2, wc3 = st.columns(3)
                with wc1:
                    wc_exposure = st.slider(
                        "Exposure (alpha)", min_value=0.5, max_value=3.0,
                        value=DEFAULT_EXPOSURE, step=0.05, key="wc_exposure",
                        help="Kecerahan setelah inversi. >1 = terang, <1 = gelap.",
                    )
                with wc2:
                    wc_sharpen = st.slider(
                        "Sharpening", min_value=1.0, max_value=5.0,
                        value=DEFAULT_SHARPEN_AMT, step=0.1, key="wc_sharpen",
                        help="Intensitas Unsharp Mask.",
                    )
                with wc3:
                    wc_threshold = st.slider(
                        "Threshold Defek (%)", min_value=30.0, max_value=99.0,
                        value=DEFAULT_CONF_THRESH, step=1.0, key="wc_threshold",
                        help="Confidence di bawah ini → dianggap NORMAL.",
                    )
                st.caption(
                    f"Pipeline aktif: BGR → Grayscale → Inversi → "
                    f"Exposure x{wc_exposure:.2f} → "
                    f"Sharpen x{wc_sharpen:.1f} → "
                    f"Crop → Model → Threshold {wc_threshold:.0f}%"
                )

                st.divider()
                rb_col, rb_info = st.columns([1, 3])
                with rb_col:
                    restart_clicked = st.button(
                        "Restart Stream",
                        key="btn_restart_stream",
                        use_container_width=True,
                        help="Matikan lalu nyalakan ulang kamera secara otomatis.",
                    )
                with rb_info:
                    st.info(
                        "Setelah mengubah slider, klik **Restart Stream** "
                        "supaya nilai baru langsung diterapkan ke kamera.",
                    )
                if restart_clicked:
                    # Set flag restart agar WebRTC di-stop dulu
                    st.session_state["_restarting"] = True
                    for k in list(st.session_state.keys()):
                        if k.startswith("webcam_") or "ceramic-defect-live" in k:
                            del st.session_state[k]
                    st.rerun()

            # Alias lokal agar sisa kode tetap singkat
            exposure    = wc_exposure
            sharpen_amt = wc_sharpen
            threshold   = wc_threshold

            # ── Session state init ─────────────────────────────────────────────
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

            # ── Kontrol webcam ─────────────────────────────────────────────────
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

            # ── Live metrics placeholders ──────────────────────────────────────
            ph_metrics = st.empty()
            ph_status  = st.empty()

            # ── VideoProcessor ─────────────────────────────────────────────────
            class DefectProcessor(VideoProcessorBase):
                def __init__(self) -> None:
                    self._model      = model
                    self._classes    = classes
                    self._lock       = threading.Lock()
                    self.result      = {
                        "label": "-", "conf": 0.0, "circle": False,
                        "original_label": None, "probs": None,
                    }
                    self.last_frame_rgb = None
                    self._frame_times: deque = deque(maxlen=30)
                    self._last_save   = 0.0
                    self._last_proc   = 0.0  # FPS throttle

                # Expose exposure & sharpen via properties (dibaca tiap frame)
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

                    # ── FPS throttle: skip inference if too soon ───────────────
                    if (t0 - self._last_proc) < _MIN_FRAME_INTERVAL:
                        # Return raw frame without re-running model
                        return frame
                    self._last_proc = t0

                    res = preprocess(img_rgb,
                                     exposure=self._exposure,
                                     sharpen_amt=self._sharpen)

                    with torch.no_grad():
                        raw_label, conf, probs = predict(self._model, res["tensor"], self._classes)

                    final_label, original_label = apply_threshold(raw_label, conf, self._threshold)

                    with self._lock:
                        self.result = {
                            "label"         : final_label,
                            "conf"          : conf,
                            "circle"        : res["circle_found"],
                            "original_label": original_label,
                            "probs"         : probs,
                        }

                    # ── Auto-save ──────────────────────────────────────────
                    now = time.time()
                    auto_save_on = getattr(self, "_auto_save", False)
                    cooldown     = getattr(self, "_save_cooldown", AUTOSAVE_COOLDOWN)
                    if auto_save_on and (now - self._last_save) >= cooldown:
                        save_prediction("webcam_live", "live_frame", final_label, conf,
                                        res["circle_found"], self._exposure, self._sharpen)
                        self._last_save = now

                    # ── FPS tracking ───────────────────────────────────────
                    self._frame_times.append(time.perf_counter())

                    # ── Render output frame ────────────────────────────────
                    # Tampilkan gambar hasil crop (grayscale → BGR)
                    out_bgr = cv2.cvtColor(res["cropped"], cv2.COLOR_GRAY2BGR)

                    if show_overlay:
                        h0, w0 = img_bgr.shape[:2]
                        h1, w1 = res["cropped"].shape[:2]

                        # Lingkaran (di-scale ke ukuran cropped)
                        if res["circle_found"] and res["circle_info"]:
                            cx, cy, r = res["circle_info"]
                            sx, sy = w1 / w0, h1 / h0
                            cx2 = int(cx * sx); cy2 = int(cy * sy); r2 = int(r * min(sx, sy))
                            c_color = (0, 200, 80) if final_label == "normal" else (0, 60, 220)
                            cv2.circle(out_bgr, (cx2, cy2), r2, c_color, 2)
                            cv2.circle(out_bgr, (cx2, cy2), 5, c_color, -1)

                        # Label text
                        if not res["circle_found"]:
                            text       = "NO CIRCLE DETECTED"
                            text_color = (0, 165, 255)
                        elif final_label == "normal":
                            text       = f"NORMAL ({raw_label} {conf:.0f}%)"
                            text_color = (30, 220, 80)
                        else:
                            text       = f"{final_label.upper()}  {conf:.0f}%"
                            text_color = (0, 60, 220)

                        # Background behind text
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                        cv2.rectangle(out_bgr, (8, 8), (tw + 16, th + 20), (0, 0, 0), -1)
                        cv2.putText(out_bgr, text, (12, th + 12),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.9, text_color, 2, cv2.LINE_AA)

                        # Exposure & sharpen info (pojok kanan bawah)
                        info_txt = f"EXP x{self._exposure:.2f}  SHP x{self._sharpen:.1f}  THR {self._threshold:.0f}%"
                        cv2.putText(out_bgr, info_txt, (10, h1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                    # FPS display
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

            # ── WebRTC streamer ────────────────────────────────────────────────
            RTC_CONFIG = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            ctx = webrtc_streamer(
                key="ceramic-defect-live",
                video_processor_factory=DefectProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # ── Live stats update ──────────────────────────────────────────────
            if ctx.video_processor:
                # Sync auto-save settings into processor instance (thread-safe via simple attr)
                ctx.video_processor._auto_save    = auto_save
                ctx.video_processor._save_cooldown = save_cd

                res = ctx.video_processor.result
                fps = ctx.video_processor.get_fps()

                # Update session counter
                lbl = res["label"]
                if lbl != "-":
                    st.session_state.webcam_session_counts[lbl] = \
                        st.session_state.webcam_session_counts.get(lbl, 0) + 1
                    st.session_state.webcam_session_total += 1

                # Metric cards
                with ph_metrics.container():
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("FPS", f"{fps:.1f}")
                    mc2.metric("Label", res["label"].upper() if res["label"] != "-" else "—")
                    mc3.metric("Confidence", f"{res['conf']:.1f}%" if res["conf"] > 0 else "—")
                    mc4.metric("Lingkaran", "Terdeteksi" if res["circle"] else "Tidak")

                # Status
                with ph_status.container():
                    if not res["circle"]:
                        st.error("Lingkaran piring tidak terdeteksi — arahkan kamera ke piring keramik.")
                    elif res["label"] == "normal":
                        st.success(
                            f"NORMAL — Deteksi '{res['original_label']}' "
                            f"({res['conf']:.1f}%) di bawah threshold {threshold:.0f}%"
                        )
                    elif res["label"] != "-":
                        st.error(f"DEFEK: {res['label'].upper()} — Confidence {res['conf']:.1f}%")

                    # Session stats mini
                    if st.session_state.webcam_session_total > 0:
                        parts = [f"`{k}`: {v}" for k, v in st.session_state.webcam_session_counts.items()]
                        st.caption(f"Session frames: {st.session_state.webcam_session_total} | " + " | ".join(parts))

                # ── Tombol aksi ────────────────────────────────────────────────
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
                        save_prediction("webcam", "live_frame",
                                        res["label"], res["conf"], res["circle"],
                                        exposure, sharpen_amt)
                        st.success(f"Disimpan: {res['label']} ({res['conf']:.1f}%)")
                with col_reset:
                    if st.button("Reset Session Stats"):
                        st.session_state.webcam_session_counts = {}
                        st.session_state.webcam_session_total  = 0
                        st.rerun()

                # Auto-save status indicator
                if auto_save:
                    st.info(f"Auto-save aktif setiap {save_cd:.0f} detik (hanya prediksi valid).")

            # ── Analisis captured frame ────────────────────────────────────────
            if st.session_state.webcam_capture is not None:
                st.divider()
                st.markdown("### Analisis Frame Terakhir")

                cap_img    = st.session_state.webcam_capture
                with st.spinner("Memproses frame…"):
                    cap_result                   = preprocess(cap_img, exposure=exposure, sharpen_amt=sharpen_amt)
                    cap_raw, cap_conf, cap_probs = predict(model, cap_result["tensor"], classes)
                    cap_final, cap_orig          = apply_threshold(cap_raw, cap_conf, threshold)

                show_pipeline_images(cap_img, cap_result)
                st.divider()
                show_result(cap_final, cap_conf, cap_probs, classes,
                            cap_result["circle_found"], cap_orig, threshold)

                col_sv, col_cl = st.columns(2)
                with col_sv:
                    if st.button("Simpan Hasil Capture"):
                        save_prediction("webcam_capture", "captured_frame",
                                        cap_final, cap_conf, cap_result["circle_found"],
                                        exposure, sharpen_amt)
                        st.success("Prediksi capture disimpan.")
                with col_cl:
                    if st.button("Hapus Capture"):
                        st.session_state.webcam_capture = None
                        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_history:
        st.subheader("Riwayat Prediksi (50 terakhir)")

        col_r, col_c = st.columns(2)
        with col_r:
            if st.button("Refresh"):
                st.rerun()
        with col_c:
            if st.button("Hapus Semua"):
                clear_history()
                st.success("Riwayat dihapus.")
                st.rerun()

        rows = get_history()
        if rows:
            # Filter
            all_labels = sorted({r[3] for r in rows})
            filter_label = st.multiselect("Filter Label", options=all_labels, default=all_labels)
            filtered = [r for r in rows if r[3] in filter_label]

            st.table([
                {
                    "ID"           : r[0],
                    "Source"       : r[1],
                    "File"         : r[2],
                    "Label"        : r[3],
                    "Conf (%)"     : f"{r[4]:.2f}",
                    "Circle"       : "Ya" if r[5] else "Tidak",
                    "Exposure"     : f"x{r[6]:.2f}" if r[6] else "-",
                    "Sharpen"      : f"x{r[7]:.1f}" if r[7] else "-",
                    "Waktu"        : r[8],
                }
                for r in filtered
            ])
        else:
            st.info("Belum ada prediksi tersimpan.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — ANALITIK
    # ══════════════════════════════════════════════════════════════════════════
    with tab_analytics:
        st.subheader("Analitik Prediksi")

        if st.button("Refresh Analitik"):
            st.rerun()

        stats = get_stats()

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Prediksi", stats["total"])
        m2.metric("Rata-rata Confidence", f"{stats['avg_conf']:.1f}%")
        m3.metric("Jumlah Kelas Terdeteksi", len(stats["by_label"]))

        if stats["by_label"]:
            st.divider()
            st.markdown("#### Distribusi Label")

            # Bar chart sederhana via st.bar_chart
            label_data = {row[0]: row[1] for row in stats["by_label"]}
            st.bar_chart(label_data)

            st.markdown("#### Detail per Label")
            total = stats["total"] or 1
            for lbl, cnt in stats["by_label"]:
                pct = cnt / total * 100
                st.progress(pct / 100, text=f"{lbl}: {cnt} prediksi ({pct:.1f}%)")
        else:
            st.info("Belum ada data untuk dianalisis. Jalankan beberapa prediksi terlebih dahulu.")

        # Tabel confidence per label dari riwayat
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
                    "Label"  : lbl,
                    "Count"  : len(confs),
                    "Min %"  : f"{min(confs):.1f}",
                    "Max %"  : f"{max(confs):.1f}",
                    "Avg %"  : f"{sum(confs)/len(confs):.1f}",
                })
            st.table(stat_rows)


if __name__ == "__main__":
    main()