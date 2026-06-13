"""
Real-time Waste Classification menggunakan MobileNetV2
Model: final_mobilenetv2_waste.keras
Dependency: tensorflow, opencv-python, numpy

Install:
    pip install tensorflow opencv-python numpy
"""

import cv2
import numpy as np
import tensorflow as tf
import time

# ── Konfigurasi ──────────────────────────────────────────────────────────────
MODEL_PATH   = r"C:\Yoga\model\final_mobilenetv2.keras" # sesuaikan path jika perlu
IMG_SIZE     = (224, 224)                          # input size MobileNetV2
CONF_THRESH  = 0.60                                # threshold confidence minimum

# Label sesuai urutan kelas saat training
CLASS_NAMES  = ["glass", "metal", "organic", "paper", "plastic"]

# Warna per kelas (BGR)
CLASS_COLORS = {
    "glass":     (255, 255,   0),   # cyan
    "metal":     (180, 180, 180),   # silver
    "organic":   (0,   200,   0),   # green
    "paper":     (255, 255, 255),   # white
    "plastic":   (0,   0,   255),   # red
}

# ── Load Model ────────────────────────────────────────────────────────────────
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary(print_fn=lambda x: None)   # silent summary
print("[INFO] Model loaded ✓")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(frame):
    """Resize + normalisasi frame untuk input MobileNetV2."""
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)          # (1, 224, 224, 3)
    return img

# ── Visualisasi ───────────────────────────────────────────────────────────────
def draw_overlay(frame, label, confidence, fps):
    h, w = frame.shape[:2]
    color = CLASS_COLORS.get(label, (255, 255, 255))

    # ── Kotak semi-transparan di bawah ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Label & confidence ──
    if confidence >= CONF_THRESH:
        text  = f"{label.upper()}  {confidence*100:.1f}%"
    else:
        text  = f"Tidak yakin  {confidence*100:.1f}%"
        color = (100, 100, 100)

    cv2.putText(frame, text, (15, h - 30),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

    # ── Bar confidence ──
    bar_w = int((w - 30) * confidence)
    cv2.rectangle(frame, (15, h - 15), (w - 15, h - 5), (60, 60, 60), -1)
    cv2.rectangle(frame, (15, h - 15), (15 + bar_w, h - 5), color, -1)

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Kotak tengah panduan ROI ──
    cx, cy = w // 2, h // 2
    roi = 200
    cv2.rectangle(frame,
                  (cx - roi, cy - roi), (cx + roi, cy + roi),
                  color, 2)
    cv2.putText(frame, "Arahkan objek ke sini", (cx - roi, cy - roi - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame

# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)           # 0 = webcam default; ganti jika perlu
    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka. Cek index kamera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Tekan  Q  untuk keluar.")

    prev_time  = time.time()
    label      = "..."
    confidence = 0.0
    frame_skip = 0   # hitung frame untuk inferensi setiap N frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame tidak terbaca, skip.")
            continue

        # Inferensi setiap 3 frame agar lebih ringan
        # Ganti bagian inferensi di dalam while loop:
        if frame_skip % 3 == 0:
            preds      = model.predict(preprocess(frame), verbose=0)[0]
            idx        = int(np.argmax(preds))
            confidence = float(preds[idx])
            # Guard: pastikan idx tidak melebihi jumlah kelas
            if idx < len(CLASS_NAMES):
                label = CLASS_NAMES[idx]
            else:
                label = f"kelas-{idx}"
            
        frame_skip += 1

        # Hitung FPS
        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Gambar overlay
        frame = draw_overlay(frame, label, confidence, fps)

        cv2.imshow("Waste Classifier — Tekan Q untuk keluar", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")

if __name__ == "__main__":
    main()