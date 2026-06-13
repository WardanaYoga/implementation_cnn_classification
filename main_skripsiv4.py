import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from collections import deque, Counter
from datetime import datetime
import os
import time

# ========================
# KONFIGURASI
# ========================
MODEL_PATH = "mobilenetv2model.tflite"
IMG_SIZE = 224
LABELS = ["glass", "metal", "organic", "paper", "plastic"]

CONF_THRESHOLD = 0.85       # Minimum confidence untuk dianggap valid
SAVE_THRESHOLD = 0.90       # Minimum confidence untuk auto-save
ENTROPY_THRESHOLD = 0.70    # Maksimum normalized entropy (lebih tinggi = lebih ragu)
CONSISTENCY_THRESHOLD = 0.70  # Minimum konsistensi voting (70% dari history)
MIN_HISTORY_FRAMES = 5      # Minimum frame sebelum prediksi dianggap stabil
STD_DEV_THRESHOLD = 20      # Minimum std deviation ROI (deteksi ada/tidaknya objek)

prediction_history = deque(maxlen=10)
prev_time = time.time()

os.makedirs("hasil_klasifikasi", exist_ok=True)

# ========================
# LOAD MODEL TFLITE
# ========================
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model berhasil dimuat")
    print(f"📊 Input shape: {input_details[0]['shape']}")
    print(f"📊 Output shape: {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ========================
# INISIALISASI KAMERA
# ========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera tidak bisa dibuka")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ========================
# INISIALISASI GUI
# ========================
root = tk.Tk()
root.title("Sistem Klasifikasi Sampah - TFLite")
root.geometry("950x600")
root.configure(bg="#1e1e1e")
root.resizable(False, False)

title = Label(root, text="SISTEM KLASIFIKASI SAMPAH",
              font=("Arial", 20, "bold"),
              fg="white", bg="#1e1e1e")
title.pack(pady=15)

main_frame = Frame(root, bg="#1e1e1e")
main_frame.pack()

# Frame kamera (kiri)
camera_frame = Frame(main_frame, bg="black", width=480, height=360,
                     relief=tk.SUNKEN, borderwidth=3)
camera_frame.grid(row=0, column=0, padx=15, pady=10)
camera_frame.pack_propagate(False)

camera_label = Label(camera_frame, bg="black")
camera_label.pack(fill=tk.BOTH, expand=True)

# Frame info (kanan)
info_frame = Frame(main_frame, bg="#2b2b2b", width=380, height=360,
                   relief=tk.RAISED, borderwidth=3)
info_frame.grid(row=0, column=1, padx=15, pady=10)
info_frame.pack_propagate(False)

lbl_title = Label(info_frame, text="HASIL DETEKSI",
                  font=("Arial", 18, "bold"),
                  fg="white", bg="#2b2b2b")
lbl_title.pack(pady=15)

result_frame = Frame(info_frame, bg="#363636", relief=tk.GROOVE, borderwidth=2)
result_frame.pack(pady=5, padx=20, fill=tk.BOTH)

lbl_class = Label(result_frame, text="---",
                  font=("Arial", 28, "bold"),
                  fg="cyan", bg="#363636",
                  height=2)
lbl_class.pack(pady=10)

lbl_conf = Label(info_frame, text="Confidence: -",
                 font=("Arial", 14),
                 fg="white", bg="#2b2b2b")
lbl_conf.pack(pady=5)

lbl_entropy = Label(info_frame, text="Entropy: -",
                    font=("Arial", 12),
                    fg="lightgray", bg="#2b2b2b")
lbl_entropy.pack(pady=3)

lbl_consistency = Label(info_frame, text="Konsistensi: -",
                        font=("Arial", 12),
                        fg="lightgray", bg="#2b2b2b")
lbl_consistency.pack(pady=3)

lbl_status = Label(info_frame, text="🟢 Kamera Aktif",
                   font=("Arial", 12),
                   fg="lime", bg="#2b2b2b")
lbl_status.pack(pady=8)

# Frame tombol
button_frame = Frame(root, bg="#1e1e1e")
button_frame.pack(pady=10)

freeze_detection = False

def toggle_detection():
    global freeze_detection
    freeze_detection = not freeze_detection
    if freeze_detection:
        btn_freeze.config(text="LANJUTKAN", bg="orange")
        lbl_status.config(text="⏸️ Deteksi Dijeda", fg="orange")
    else:
        btn_freeze.config(text="JEDA", bg="blue")
        lbl_status.config(text="🟢 Kamera Aktif", fg="lime")

def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

btn_freeze = Button(button_frame, text="JEDA",
                    font=("Arial", 12, "bold"),
                    bg="blue", fg="white",
                    width=12, height=2,
                    command=toggle_detection)
btn_freeze.grid(row=0, column=0, padx=10)

btn_exit = Button(button_frame, text="KELUAR",
                  font=("Arial", 12, "bold"),
                  bg="red", fg="white",
                  width=12, height=2,
                  command=exit_app)
btn_exit.grid(row=0, column=1, padx=10)

# ========================
# WARNA PER KELAS
# ========================
CLASS_COLORS = {
    "glass":   "cyan",
    "metal":   "lightgray",
    "organic": "lime",
    "paper":   "orange",
    "plastic": "yellow"
}

# ========================
# FUNGSI PREPROCESSING
# ========================
def preprocess_image(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    # Manual preprocess_input MobileNetV2 (scale ke -1 ~ 1)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img

# ========================
# FUNGSI KLASIFIKASI
# ========================
def classify_image(frame):
    try:
        img = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        if len(output) != len(LABELS):
            print(f"⚠️ Output size ({len(output)}) != Labels size ({len(LABELS)})")
            return "ERROR", 0.0, None

        class_id = np.argmax(output)
        confidence = float(output[class_id])
        label = LABELS[class_id]
        return label, confidence, output

    except Exception as e:
        print(f"❌ Error klasifikasi: {e}")
        return "ERROR", 0.0, None

# ========================
# FUNGSI DETEKSI OBJEK (ROI)
# ========================
def has_object_in_roi(roi):
    """
    Cek apakah ROI mengandung objek berdasarkan variasi warna.
    Latar polos/kosong memiliki std deviation rendah.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    return std_dev > STD_DEV_THRESHOLD, float(std_dev)

# ========================
# FUNGSI CEK ENTROPY
# ========================
def compute_entropy(output):
    """
    Hitung normalized entropy dari distribusi probabilitas output model.
    Entropy tinggi = model tidak yakin / distribusi merata.
    Entropy rendah = model yakin pada satu kelas.
    Return: normalized_entropy (0.0 ~ 1.0)
    """
    probs = np.array(output, dtype=np.float64)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum()  # pastikan sum = 1
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(LABELS))
    normalized_entropy = entropy / max_entropy
    return float(normalized_entropy)

# ========================
# FUNGSI VOTING STABIL
# ========================
def get_stable_prediction(history, current_conf):
    """
    Kembalikan prediksi stabil berdasarkan voting history.
    Dianggap stabil jika:
    - History sudah cukup (>= MIN_HISTORY_FRAMES)
    - Kelas dominan muncul >= CONSISTENCY_THRESHOLD
    - Confidence saat ini >= CONF_THRESHOLD
    """
    if len(history) < MIN_HISTORY_FRAMES:
        return None, 0.0

    most_common_label, count = Counter(history).most_common(1)[0]
    consistency = count / len(history)

    if consistency >= CONSISTENCY_THRESHOLD and current_conf >= CONF_THRESHOLD:
        return most_common_label, consistency

    return None, consistency

# ========================
# FUNGSI UPDATE FRAME
# ========================
def update_frame():
    global prev_time

    ret, frame = cap.read()

    if not ret:
        lbl_status.config(text="❌ Kamera Error", fg="red")
        root.after(100, update_frame)
        return

    # Hitung FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time + 1e-9)
    prev_time = current_time

    if not freeze_detection:
        h, w, _ = frame.shape
        x1 = int(w * 0.25)
        y1 = int(h * 0.20)
        x2 = int(w * 0.75)
        y2 = int(h * 0.80)
        roi = frame[y1:y2, x1:x2]

        # --- CEK 1: Ada objek di ROI? ---
        object_detected, std_dev = has_object_in_roi(roi)

        if not object_detected:
            # Tidak ada objek → reset semua
            prediction_history.clear()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(frame, "Tidak ada objek",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            lbl_class.config(text="TIDAK ADA OBJEK", fg="gray")
            lbl_conf.config(text="Confidence: -")
            lbl_entropy.config(text=f"Std Dev ROI: {std_dev:.1f}")
            lbl_consistency.config(text="Konsistensi: -")

        else:
            # Ada objek → lanjut klasifikasi
            label, confidence, output = classify_image(roi)

            if label != "ERROR" and output is not None:

                # --- CEK 2: Entropy check ---
                entropy = compute_entropy(output)
                is_uncertain = entropy > ENTROPY_THRESHOLD

                # --- CEK 3: Voting stabil ---
                prediction_history.append(label)
                stable_label, consistency = get_stable_prediction(
                    prediction_history, confidence
                )

                # Update info panel
                lbl_entropy.config(text=f"Entropy: {entropy:.3f} {'⚠️' if is_uncertain else '✅'}")
                lbl_consistency.config(text=f"Konsistensi: {consistency*100:.0f}%")
                lbl_conf.config(text=f"Confidence: {confidence*100:.2f}%")

                # Tentukan warna kotak ROI dan status
                if is_uncertain:
                    # Entropy tinggi → model ragu
                    box_color = (0, 165, 255)  # orange
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, "Tidak Yakin",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    lbl_class.config(text="TIDAK YAKIN", fg="yellow")
                    prediction_history.clear()

                elif stable_label is None:
                    # Belum stabil → tunggu lebih banyak frame
                    box_color = (0, 255, 255)  # kuning
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"Mendeteksi... ({label})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    lbl_class.config(text="MENDETEKSI...", fg="yellow")

                else:
                    # Prediksi stabil dan confident
                    color_hex = CLASS_COLORS.get(stable_label, "white")
                    color_bgr_map = {
                        "cyan":      (255, 255, 0),
                        "lightgray": (200, 200, 200),
                        "lime":      (0, 255, 0),
                        "orange":    (0, 165, 255),
                        "yellow":    (0, 255, 255),
                        "white":     (255, 255, 255),
                    }
                    box_color = color_bgr_map.get(color_hex, (0, 255, 0))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(
                        frame,
                        f"{stable_label.upper()} ({confidence*100:.1f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2
                    )
                    lbl_class.config(text=stable_label.upper(), fg=color_hex)

                    # Auto-save jika confidence sangat tinggi
                    if confidence >= SAVE_THRESHOLD:
                        filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = f"hasil_klasifikasi/{stable_label}_{filename}.jpg"
                        cv2.imwrite(save_path, frame)
                        print(f"💾 Tersimpan: {save_path}")

                # Debug print semua kelas
                print(f"\n=== Frame | FPS: {fps:.1f} | Entropy: {entropy:.3f} ===")
                for i, score in enumerate(output):
                    bar = "█" * int(score * 20)
                    print(f"  {LABELS[i]:10s}: {score*100:.1f}% {bar}")

            else:
                lbl_class.config(text="ERROR", fg="red")

    # Overlay FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan frame di GUI
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (480, 360))
    img_pil = Image.fromarray(frame_resized)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    camera_label.imgtk = img_tk
    camera_label.configure(image=img_tk)

    root.after(10, update_frame)

# ========================
# JALANKAN APLIKASI
# ========================
print("🚀 Aplikasi dimulai...")
print(f"   CONF_THRESHOLD    : {CONF_THRESHOLD}")
print(f"   ENTROPY_THRESHOLD : {ENTROPY_THRESHOLD}")
print(f"   CONSISTENCY       : {CONSISTENCY_THRESHOLD}")
print(f"   STD_DEV_THRESHOLD : {STD_DEV_THRESHOLD}")
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
print("✅ Aplikasi ditutup")