import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button, Frame, ttk
from PIL import Image, ImageTk
from collections import deque, Counter
from datetime import datetime
import os
import time
import csv

# ========================
# KONFIGURASI
# ========================
MODEL_PATH = "newfinalmobilenetv2waste.tflite"
IMG_SIZE = 224
LABELS = ["glass", "metal", "organic", "paper", "plastic"]

CONF_THRESHOLD      = 0.85
ENTROPY_THRESHOLD   = 0.70
CONSISTENCY_THRESHOLD = 0.70
MIN_HISTORY_FRAMES  = 5
STD_DEV_THRESHOLD   = 20
COUNTDOWN_SECONDS   = 3      # Hitung mundur sebelum deteksi dimulai

prediction_history  = deque(maxlen=10)
prev_time           = time.time()

os.makedirs("hasil_klasifikasi", exist_ok=True)

# CSV log
CSV_PATH = "hasil_klasifikasi/log_deteksi.csv"
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["No", "Waktu", "Label", "Confidence", "Entropy", "File"])

# Baca jumlah data yang sudah ada
with open(CSV_PATH, "r") as f:
    detection_count = sum(1 for _ in f) - 1  # minus header

# ========================
# LOAD MODEL TFLITE
# ========================
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model berhasil dimuat")
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
# STATE MESIN
# ========================
# IDLE       → Menunggu pengguna menekan MULAI
# COUNTDOWN  → Hitung mundur sebelum deteksi
# DETECTING  → Kamera aktif, sedang mendeteksi
# DONE       → Hasil ditemukan, disimpan, menunggu reset
STATE = "IDLE"
countdown_value  = COUNTDOWN_SECONDS
countdown_start  = None
last_saved_label = None
last_saved_conf  = None
last_saved_file  = None

# ========================
# INISIALISASI GUI
# ========================
root = tk.Tk()
root.title("Sistem Klasifikasi Sampah - Mode Manual")
root.geometry("950x650")
root.configure(bg="#1e1e1e")
root.resizable(False, False)

title = Label(root, text="SISTEM KLASIFIKASI SAMPAH",
              font=("Arial", 20, "bold"),
              fg="white", bg="#1e1e1e")
title.pack(pady=10)

main_frame = Frame(root, bg="#1e1e1e")
main_frame.pack()

# --- Kamera (kiri) ---
camera_frame = Frame(main_frame, bg="black", width=480, height=360,
                     relief=tk.SUNKEN, borderwidth=3)
camera_frame.grid(row=0, column=0, padx=15, pady=10)
camera_frame.pack_propagate(False)

camera_label = Label(camera_frame, bg="black")
camera_label.pack(fill=tk.BOTH, expand=True)

# --- Info (kanan) ---
info_frame = Frame(main_frame, bg="#2b2b2b", width=380, height=360,
                   relief=tk.RAISED, borderwidth=3)
info_frame.grid(row=0, column=1, padx=15, pady=10)
info_frame.pack_propagate(False)

lbl_info_title = Label(info_frame, text="HASIL DETEKSI",
                       font=("Arial", 16, "bold"),
                       fg="white", bg="#2b2b2b")
lbl_info_title.pack(pady=10)

result_frame = Frame(info_frame, bg="#363636", relief=tk.GROOVE, borderwidth=2)
result_frame.pack(pady=5, padx=15, fill=tk.BOTH)

lbl_class = Label(result_frame, text="---",
                  font=("Arial", 30, "bold"),
                  fg="gray", bg="#363636", height=2)
lbl_class.pack(pady=10)

lbl_conf        = Label(info_frame, text="Confidence: -",
                        font=("Arial", 13), fg="white", bg="#2b2b2b")
lbl_conf.pack(pady=3)

lbl_entropy     = Label(info_frame, text="Entropy: -",
                        font=("Arial", 11), fg="lightgray", bg="#2b2b2b")
lbl_entropy.pack(pady=2)

lbl_consistency = Label(info_frame, text="Konsistensi: -",
                        font=("Arial", 11), fg="lightgray", bg="#2b2b2b")
lbl_consistency.pack(pady=2)

lbl_count       = Label(info_frame, text=f"Total Tersimpan: {detection_count}",
                        font=("Arial", 11), fg="lightblue", bg="#2b2b2b")
lbl_count.pack(pady=5)

lbl_status      = Label(info_frame, text="⏳ Siapkan sampah, lalu tekan MULAI",
                        font=("Arial", 11), fg="orange", bg="#2b2b2b",
                        wraplength=340, justify="center")
lbl_status.pack(pady=8)

# --- Tombol ---
button_frame = Frame(root, bg="#1e1e1e")
button_frame.pack(pady=8)

def on_start():
    global STATE, countdown_start, prediction_history
    if STATE == "IDLE":
        STATE = "COUNTDOWN"
        countdown_start = time.time()
        prediction_history.clear()
        btn_start.config(state=tk.DISABLED, bg="#555")
        btn_reset.config(state=tk.NORMAL, bg="orange")
        lbl_status.config(text="🔄 Bersiap...", fg="yellow")

def on_reset():
    global STATE, last_saved_label, last_saved_conf, last_saved_file
    STATE = "IDLE"
    prediction_history.clear()
    last_saved_label = None
    last_saved_conf  = None
    last_saved_file  = None
    btn_start.config(state=tk.NORMAL, bg="green", text="▶ MULAI DETEKSI")
    btn_reset.config(state=tk.DISABLED, bg="#555")
    lbl_class.config(text="---", fg="gray")
    lbl_conf.config(text="Confidence: -")
    lbl_entropy.config(text="Entropy: -")
    lbl_consistency.config(text="Konsistensi: -")
    lbl_status.config(text="⏳ Siapkan sampah, lalu tekan MULAI", fg="orange")

def on_exit():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

btn_start = Button(button_frame, text="▶ MULAI DETEKSI",
                   font=("Arial", 13, "bold"),
                   bg="green", fg="white",
                   width=16, height=2,
                   command=on_start)
btn_start.grid(row=0, column=0, padx=10)

btn_reset = Button(button_frame, text="🔄 ULANGI",
                   font=("Arial", 13, "bold"),
                   bg="#555", fg="white",
                   width=12, height=2,
                   state=tk.DISABLED,
                   command=on_reset)
btn_reset.grid(row=0, column=1, padx=10)

btn_exit = Button(button_frame, text="✖ KELUAR",
                  font=("Arial", 13, "bold"),
                  bg="red", fg="white",
                  width=12, height=2,
                  command=on_exit)
btn_exit.grid(row=0, column=2, padx=10)

# ========================
# WARNA PER KELAS
# ========================
CLASS_COLORS_TK = {
    "glass":   "cyan",
    "metal":   "lightgray",
    "organic": "lime",
    "paper":   "orange",
    "plastic": "yellow",
}
CLASS_COLORS_BGR = {
    "glass":   (255, 255, 0),
    "metal":   (200, 200, 200),
    "organic": (0, 255, 0),
    "paper":   (0, 165, 255),
    "plastic": (0, 255, 255),
}

# ========================
# FUNGSI HELPER
# ========================
def preprocess_image(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(frame):
    try:
        img = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        if len(output) != len(LABELS):
            return "ERROR", 0.0, None
        class_id   = np.argmax(output)
        confidence = float(output[class_id])
        label      = LABELS[class_id]
        return label, confidence, output
    except Exception as e:
        print(f"❌ Error klasifikasi: {e}")
        return "ERROR", 0.0, None

def has_object_in_roi(roi):
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    return std_dev > STD_DEV_THRESHOLD, float(std_dev)

def compute_entropy(output):
    probs = np.array(output, dtype=np.float64)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum()
    entropy     = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(LABELS))
    return float(entropy / max_entropy)

def get_stable_prediction(history, current_conf):
    if len(history) < MIN_HISTORY_FRAMES:
        return None, 0.0
    most_common_label, count = Counter(history).most_common(1)[0]
    consistency = count / len(history)
    if consistency >= CONSISTENCY_THRESHOLD and current_conf >= CONF_THRESHOLD:
        return most_common_label, consistency
    return None, consistency

def save_result(frame, label, confidence, entropy):
    global detection_count
    detection_count += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"hasil_klasifikasi/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            detection_count,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{confidence*100:.2f}%",
            f"{entropy:.3f}",
            filename
        ])

    lbl_count.config(text=f"Total Tersimpan: {detection_count}")
    print(f"💾 Disimpan: {filename}")
    return filename

# ========================
# FUNGSI UPDATE FRAME
# ========================
def update_frame():
    global STATE, countdown_value, prev_time
    global last_saved_label, last_saved_conf, last_saved_file

    ret, frame = cap.read()
    if not ret:
        lbl_status.config(text="❌ Kamera Error", fg="red")
        root.after(100, update_frame)
        return

    current_time = time.time()
    fps          = 1.0 / (current_time - prev_time + 1e-9)
    prev_time    = current_time

    h, w, _ = frame.shape
    x1 = int(w * 0.25)
    y1 = int(h * 0.20)
    x2 = int(w * 0.75)
    y2 = int(h * 0.80)
    roi = frame[y1:y2, x1:x2]

    # ========================
    # STATE: IDLE
    # ========================
    if STATE == "IDLE":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(frame, "Siapkan sampah di sini",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # ========================
    # STATE: COUNTDOWN
    # ========================
    elif STATE == "COUNTDOWN":
        elapsed   = current_time - countdown_start
        remaining = COUNTDOWN_SECONDS - int(elapsed)

        if remaining <= 0:
            STATE = "DETECTING"
            lbl_status.config(text="🔍 Mendeteksi...", fg="lime")
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Mulai dalam {remaining}...",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # Overlay countdown besar di tengah frame
            cv2.putText(frame, str(remaining),
                        (w // 2 - 30, h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8)
            lbl_status.config(text=f"⏱️ Mulai dalam {remaining} detik...", fg="yellow")

    # ========================
    # STATE: DETECTING
    # ========================
    elif STATE == "DETECTING":
        object_found, std_dev = has_object_in_roi(roi)

        if not object_found:
            prediction_history.clear()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(frame, "Tidak ada objek",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            lbl_class.config(text="TIDAK ADA OBJEK", fg="gray")
            lbl_conf.config(text="Confidence: -")
            lbl_status.config(text="⚠️ Objek tidak terdeteksi di area", fg="orange")

        else:
            label, confidence, output = classify_image(roi)

            if label != "ERROR" and output is not None:
                entropy = compute_entropy(output)
                prediction_history.append(label)
                stable_label, consistency = get_stable_prediction(prediction_history, confidence)

                lbl_entropy.config(
                    text=f"Entropy: {entropy:.3f} {'⚠️' if entropy > ENTROPY_THRESHOLD else '✅'}")
                lbl_consistency.config(text=f"Konsistensi: {consistency*100:.0f}%")
                lbl_conf.config(text=f"Confidence: {confidence*100:.2f}%")

                if entropy > ENTROPY_THRESHOLD:
                    # Model tidak yakin
                    prediction_history.clear()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "Tidak Yakin",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    lbl_class.config(text="TIDAK YAKIN", fg="yellow")
                    lbl_status.config(text="🤔 Model ragu, pastikan objek jelas", fg="yellow")

                elif stable_label is None:
                    # Masih mengumpulkan frame
                    progress = len(prediction_history)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Mendeteksi... ({label})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # Progress bar di bawah ROI
                    bar_w = int((x2 - x1) * progress / MIN_HISTORY_FRAMES)
                    cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 15), (50, 50, 50), -1)
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_w, y2 + 15), (0, 255, 255), -1)
                    lbl_class.config(text=label.upper(), fg="yellow")
                    lbl_status.config(
                        text=f"🔍 Mengumpulkan data... ({progress}/{MIN_HISTORY_FRAMES})",
                        fg="yellow")

                else:
                    # ✅ Hasil stabil → simpan otomatis
                    box_color = CLASS_COLORS_BGR.get(stable_label, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(frame,
                                f"{stable_label.upper()} ({confidence*100:.1f}%)",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                    # Simpan hasil
                    saved_file = save_result(frame, stable_label, confidence, entropy)

                    last_saved_label = stable_label
                    last_saved_conf  = confidence
                    last_saved_file  = saved_file

                    tk_color = CLASS_COLORS_TK.get(stable_label, "white")
                    lbl_class.config(text=stable_label.upper(), fg=tk_color)
                    lbl_status.config(
                        text=f"✅ Tersimpan! Tekan ULANGI untuk deteksi berikutnya",
                        fg="lime")
                    btn_start.config(state=tk.DISABLED, bg="#555",
                                     text="▶ MULAI DETEKSI")

                    STATE = "DONE"

    # ========================
    # STATE: DONE
    # ========================
    elif STATE == "DONE":
        if last_saved_label:
            box_color = CLASS_COLORS_BGR.get(last_saved_label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(frame,
                        f"✅ {last_saved_label.upper()} - TERSIMPAN",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Overlay FPS
    cv2.putText(frame, f"FPS: {fps:.1f} | {STATE}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan ke GUI
    frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (480, 360))
    img_pil       = Image.fromarray(frame_resized)
    img_tk        = ImageTk.PhotoImage(image=img_pil)
    camera_label.imgtk = img_tk
    camera_label.configure(image=img_tk)

    root.after(10, update_frame)

# ========================
# JALANKAN
# ========================
print("🚀 Aplikasi dimulai (Mode Manual)")
print("   Alur: Taruh sampah → MULAI → Hitung mundur → Deteksi → Simpan → ULANGI")
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
print("✅ Aplikasi ditutup")