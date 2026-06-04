---
title: Ceramic Tile Defect Detector
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# 🔍 Ceramic Tile Defect Detector

Aplikasi deteksi defek pada piring/ubin keramik menggunakan **DINOv2 ViT-Small** (fine-tuned).

## Kelas Defek
| Kelas     | Deskripsi                    |
|-----------|------------------------------|
| `crack`   | Retakan pada permukaan       |
| `spot`    | Noda / bercak                |
| `pinhole` | Lubang kecil (pin hole)      |

## Pipeline Pemrosesan Gambar
```
Input (RGB)
   ↓
Grayscale (0–255)
   ↓
Unsharp Mask (sharpen)
   ↓
Replicate → 3-channel
   ↓
Resize 224×224 + Normalize
   ↓
DINOv2 ViT-Small Head
   ↓
Softmax → Label + Confidence
```

## Fitur
- **Upload Gambar** — prediksi dari file JPG/PNG
- **Webcam Live** — deteksi real-time lewat kamera, overlay label di video
- **Riwayat** — semua prediksi disimpan di SQLite lokal

## Model
- Arsitektur: `vit_small_patch14_dinov2.lvd142m` via `timm`
- Backbone: Frozen (hanya classifier head yang dilatih)
- Input size: 224×224
- File: `best_model_dinov2.pth`
