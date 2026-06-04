"""
Ceramic Tile Defect Detector
Backend-first Streamlit app
Pipeline: Image/Webcam → Grayscale → (Detect Circle) → Invert → Exposure +30% → Sharpen → Crop (Apply Mask) → DINOv2 → Prediction → Threshold 70% → SQLite
"""

import os
import sqlite3
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

# ─────────────────────────────────────────────────────────────────────────────
CLASSES_2 = ["crack", "spot"]
CLASSES_3 = ["crack", "pinhole", "spot"]

# THRESHOLD: Jika prediksi crack/spot di bawah nilai ini, dianggap "normal"
CONFIDENCE_THRESHOLD = 70.0  # Dalam persen (70%)
DEFECT_CLASSES_FOR_THRESHOLD = ["crack", "spot"]
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL,
            filename    TEXT,
            label       TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            circle_found INTEGER DEFAULT 0,
            created_at  TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(source: str, filename: str, label: str,
                    confidence: float, circle_found: bool = False) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions "
        "(source, filename, label, confidence, circle_found, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source, filename or "-", label, round(confidence, 4),
         int(circle_found), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 30) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, source, filename, label, confidence, circle_found, created_at "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


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
    n  = sd["head.weight"].shape[0]
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

    classes    = _get_classes_from_checkpoint(MODEL_PATH)
    num_classes = len(classes)

    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        img_size=IMG_SIZE,
    )
    in_features = getattr(model, "num_features", None) or getattr(model, "embed_dim", None)
    model.head  = nn.Linear(in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model, classes


# ══════════════════════════════════════════════════════════════════════════════
# CIRCLE DETECTION (Hanya deteksi & buat mask, tanpa apply crop dulu)
# ══════════════════════════════════════════════════════════════════════════════
def detect_circle_mask(gray: np.ndarray) -> tuple[np.ndarray | None, bool, tuple | None]:
    """
    Deteksi lingkaran terbesar pada gambar grayscale menggunakan HoughCircles.
    Menghasilkan mask biner, tetapi BELUM meng-aplikasikannya ke gambar.
    """
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp       = 1.2,
        minDist  = min(h, w) * 0.4,
        param1   = 100,
        param2   = 40,
        minRadius= int(min(h, w) * 0.2),
        maxRadius= int(min(h, w) * 0.55),
    )

    if circles is None:
        return None, False, None

    circles = np.round(circles[0]).astype(int)
    cx, cy, r = max(circles, key=lambda c: c[2])

    # Buat circular mask
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

    return mask, True, (cx, cy, r)


def draw_circle_overlay(img_rgb: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    overlay = img_rgb.copy()
    cv2.circle(overlay, (cx, cy), r, (0, 220, 0), 3)
    cv2.circle(overlay, (cx, cy), 4, (0, 220, 0), -1)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(img_rgb: np.ndarray) -> dict:
    """
    Pipeline lengkap:
        RGB
         → Grayscale
         → (Deteksi Lingkaran & Buat Mask)
         → Inversi Warna (Seluruh gambar)
         → Exposure +30% (Seluruh gambar)
         → Unsharp Mask / Sharpen (Seluruh gambar)
         → Crop / Terapkan Mask (Luar lingkaran jadi hitam)
         → 3-channel replicate
         → Resize + Normalize (tensor)
    """
    # 1. Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Deteksi lingkaran (hanya dapatkan mask & info, gambar belum di-crop)
    mask, circle_found, circle_info = detect_circle_mask(gray)

    # 3. Inversi Warna
    inverted = cv2.bitwise_not(gray)

    # 4. Exposure +30%
    exposed = cv2.convertScaleAbs(inverted, alpha=1.3, beta=0)

    # 5. Sharpen via Unsharp Mask
    blurred   = cv2.GaussianBlur(exposed, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(exposed, 1.5, blurred, -0.5, 0)

    # 6. Crop (Terapkan mask setelah sharpen)
    if circle_found and mask is not None:
        cropped = cv2.bitwise_and(sharpened, sharpened, mask=mask)
    else:
        cropped = sharpened.copy()

    # 7. 3-channel untuk model
    rgb_3ch = cv2.merge([cropped, cropped, cropped])

    # 8. Transform ke tensor
    tensor = _transform(rgb_3ch).unsqueeze(0).to(DEVICE)

    # 9. Overlay lingkaran di gambar asli (display saja)
    annotated = img_rgb.copy()
    if circle_found and circle_info:
        cx, cy, r = circle_info
        annotated = draw_circle_overlay(img_rgb, cx, cy, r)

    return {
        "tensor"        : tensor,
        "gray"          : gray,
        "inverted"      : inverted,
        "exposed"       : exposed,
        "sharpened"     : sharpened,
        "cropped"       : cropped,
        "annotated_rgb" : annotated,
        "circle_found"  : circle_found,
        "circle_info"   : circle_info,
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
ICONS = {"crack": "🔴", "spot": "🟡", "pinhole": "🔵", "normal": "🟢"}

def apply_threshold(label: str, conf: float) -> tuple[str, str | None]:
    if label in DEFECT_CLASSES_FOR_THRESHOLD and conf < CONFIDENCE_THRESHOLD:
        return "normal", label
    return label, None


def show_result(label: str, conf: float, probs: np.ndarray,
                classes: list[str], circle_found: bool, 
                original_label: str | None = None) -> None:
    
    # Pesan khusus jika lingkaran tidak terdeteksi
    if not circle_found:
        st.error("⚠️ **Lingkaran piring tidak terdeteksi.** Kemungkinan: Piring tidak berbentuk lingkaran, piring terlalu rusak, atau objek bukanlah piring.")
        st.info("ℹ️ Model tetap dijalankan pada gambar penuh, namun hasil prediksi mungkin tidak akurat.")
    
    icon = ICONS.get(label, "⚪")
    
    if label == "normal" and original_label:
        st.markdown(f"### {icon} Prediksi: `NORMAL (TIDAK ADA DEFEK)`")
        st.info(
            f"ℹ️ Model mendeteksi indikasi **'{original_label}'** dengan confidence **{conf:.1f}%**, "
            f"namun karena di bawah threshold ({CONFIDENCE_THRESHOLD}%), piring dinyatakan **NORMAL**."
        )
    else:
        st.markdown(f"### {icon} Prediksi: `{label.upper()}` — {conf:.1f}%")
        
    for i, cls in enumerate(classes):
        st.progress(float(probs[i]), text=f"{cls}: {probs[i]*100:.1f}%")


def show_pipeline_images(orig_rgb: np.ndarray, result: dict) -> None:
    st.markdown("#### 🖼️ Pipeline Gambar")

    # Baris 1: Original, Grayscale, Inversi Warna
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(result["annotated_rgb"],
                 caption="① Original + Deteksi Lingkaran",
                 use_container_width=True)
    with col2:
        st.image(result["gray"],
                 caption="② Grayscale",
                 use_container_width=True, clamp=True)
    with col3:
        st.image(result["inverted"],
                 caption="③ Inversi Warna (Seluruh Gambar)",
                 use_container_width=True, clamp=True)

    # Baris 2: Exposure, Sharpened, Crop
    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(result["exposed"],
                 caption="④ Exposure +30%",
                 use_container_width=True, clamp=True)
    with col5:
        st.image(result["sharpened"],
                 caption="⑤ Sharpened (Seluruh Gambar)",
                 use_container_width=True, clamp=True)
    with col6:
        label_crop = ("⑥ Crop (Luar Lingkaran Hitam)" if result["circle_found"] 
                      else "⑥ Crop Gagal (Lingkaran Tidak Terdeteksi)")
        st.image(result["cropped"],
                 caption=label_crop,
                 use_container_width=True, clamp=True)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(page_title="Ceramic Defect Detector", layout="centered")
    st.title("🔍 Ceramic Tile Defect Detector")

    init_db()
    model, classes = load_model()

    st.caption(
        f"Model: DINOv2 ViT-Small · Device: {DEVICE} · "
        f"Classes ({len(classes)}): {', '.join(classes)} · "
        f"Threshold Defek: {CONFIDENCE_THRESHOLD}%"
    )

    tab_upload, tab_webcam, tab_history = st.tabs([
        "📤 Upload Gambar",
        "📷 Webcam Live",
        "📋 Riwayat Prediksi",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.subheader("Upload Gambar Keramik")
        uploaded = st.file_uploader(
            "Pilih gambar (.jpg / .jpeg / .png)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is not None:
            pil_img   = Image.open(uploaded).convert("RGB")
            img_array = np.array(pil_img)

            result             = preprocess(img_array)
            raw_label, conf, probs = predict(model, result["tensor"], classes)
            
            # Terapkan Threshold
            final_label, original_label = apply_threshold(raw_label, conf)

            st.divider()
            show_pipeline_images(img_array, result)

            st.divider()
            show_result(final_label, conf, probs, classes, 
                        result["circle_found"], original_label)

            save_prediction("upload", uploaded.name, final_label, conf, result["circle_found"])
            st.success("✅ Prediksi disimpan ke database.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — WEBCAM LIVE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_webcam:
        st.subheader("Live Webcam Detection")

        if not WEBRTC_AVAILABLE:
            st.error(
                "Paket `streamlit-webrtc` dan `av` belum terinstall.\n\n"
                "Jalankan: `pip install streamlit-webrtc av`"
            )
        else:
            st.info(
                "Pipeline per frame: **BGR → RGB → Grayscale → Inversi Warna → "
                "Exposure +30% → Sharpen → Crop → Model → Threshold**\n\n"
                "Overlay hijau = piring normal (di bawah threshold defek)."
            )

            class DefectProcessor(VideoProcessorBase):
                def __init__(self) -> None:
                    self._model   = model
                    self._classes = classes
                    self.result   = {"label": "-", "conf": 0.0, "circle": False, "original_label": None}

                def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                    img_bgr = frame.to_ndarray(format="bgr24")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    res               = preprocess(img_rgb)
                    raw_label, conf, _ = predict(self._model, res["tensor"], self._classes)
                    
                    # Terapkan Threshold
                    final_label, original_label = apply_threshold(raw_label, conf)
                    
                    self.result = {
                        "label": final_label, 
                        "conf": conf, 
                        "circle": res["circle_found"],
                        "original_label": original_label
                    }

                    # Gunakan gambar yang sudah di-crop untuk ditampilkan di webcam
                    out_bgr = cv2.cvtColor(res["cropped"], cv2.COLOR_GRAY2BGR)

                    # Tampilan teks dan lingkaran pada Webcam
                    if res["circle_found"] and res["circle_info"]:
                        cx, cy, r = res["circle_info"]
                        h0, w0 = img_bgr.shape[:2]
                        h1, w1 = res["cropped"].shape[:2]
                        sx, sy = w1 / w0, h1 / h0
                        cx2, cy2, r2 = int(cx * sx), int(cy * sy), int(r * min(sx, sy))
                        cv2.circle(out_bgr, (cx2, cy2), r2, (0, 220, 0), 2)
                        
                        if final_label == "normal":
                            text = f"NORMAL ({raw_label}: {conf:.1f}%)"
                            text_color = (0, 255, 0)  # Hijau
                        else:
                            text = f"{final_label.upper()}: {conf:.1f}%"
                            text_color = (0, 0, 255)  # Merah untuk defek
                    else:
                        text = "BUKAN PIRING / RUSAK"
                        text_color = (0, 165, 255)  # Oranye

                    cv2.putText(out_bgr, text, (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

                    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

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

            if ctx.video_processor:
                res = ctx.video_processor.result
                
                # Tampilan status di bawah webcam
                if not res["circle"]:
                    st.error("⚠️ **Lingkaran piring tidak terdeteksi.** Kemungkinan: Piring tidak berbentuk lingkaran, piring terlalu rusak, atau objek bukanlah piring.")
                else:
                    circle_txt = "✅ terdeteksi"
                    if res["label"] == "normal":
                        st.markdown(
                            f"**Live →** 🟢 `NORMAL` — Deteksi {res['original_label']} ({res['conf']:.1f}%) "
                            f"dibawah threshold | Lingkaran: {circle_txt}"
                        )
                    else:
                        st.markdown(
                            f"**Live →** `{res['label'].upper()}` — {res['conf']:.1f}%  "
                            f"| Lingkaran: {circle_txt}"
                        )
                    
                if st.button("💾 Simpan Prediksi Sekarang"):
                    save_prediction("webcam", "live_frame",
                                    res["label"], res["conf"], res["circle"])
                    st.success(f"Disimpan: {res['label']} ({res['conf']:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_history:
        st.subheader("Riwayat Prediksi (30 terakhir)")

        col_r, col_c = st.columns(2)
        with col_r:
            if st.button("🔄 Refresh"):
                st.rerun()
        with col_c:
            if st.button("🗑️ Hapus Semua"):
                clear_history()
                st.success("Riwayat dihapus.")
                st.rerun()

        rows = get_history()
        if rows:
            st.table([
                {
                    "ID"           : r[0],
                    "Source"       : r[1],
                    "File"         : r[2],
                    "Label"        : r[3],
                    "Conf (%)"     : f"{r[4]:.2f}",
                    "Circle Found" : "✅" if r[5] else "❌",
                    "Waktu"        : r[6],
                }
                for r in rows
            ])
        else:
            st.info("Belum ada prediksi tersimpan.")


if __name__ == "__main__":
    main()

    #BDCIHBHIVBOEUFBVIERBCI