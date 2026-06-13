import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from collections import deque, Counter
from keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import os
import time

# ========================
# KONFIGURASI
# ========================
MODEL_PATH = "mobilenetv2model.tflite"
IMG_SIZE = 224
LABELS = ["glass", "metal", "organic", "paper", "plastic"]
CONF_THRESHOLD = 0.70
SAVE_THRESHOLD = 0.90

prediction_history = deque(maxlen=10)

# FIX 1: prev_time harus diinisialisasi sekali di sini (bukan di tengah kode)
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
root.title("Image Classification System - TFLite")
root.geometry("950x550")
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
lbl_title.pack(pady=25)

result_frame = Frame(info_frame, bg="#363636", relief=tk.GROOVE, borderwidth=2)
result_frame.pack(pady=10, padx=20, fill=tk.BOTH)

lbl_class = Label(result_frame, text="---",
                  font=("Arial", 28, "bold"),
                  fg="cyan", bg="#363636",
                  height=2)
lbl_class.pack(pady=15)

lbl_conf = Label(info_frame, text="Confidence: 0.00%",
                 font=("Arial", 16),
                 fg="white", bg="#2b2b2b")
lbl_conf.pack(pady=15)

lbl_status = Label(info_frame, text="🟢 Kamera Aktif",
                   font=("Arial", 12),
                   fg="lime", bg="#2b2b2b")
lbl_status.pack(pady=10)

# Frame tombol
button_frame = Frame(root, bg="#1e1e1e")
button_frame.pack(pady=15)

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
    "glass": "cyan",
    "metal": "lightgray",
    "organic": "lime",
    "paper": "orange",
    "plastic": "yellow"
}

# ========================
# FUNGSI PREPROCESSING
# ========================
def preprocess_image(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
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
            print(f"⚠️ Warning: Output size ({len(output)}) != Labels size ({len(LABELS)})")
            return "ERROR", 0.0, None

        class_id = np.argmax(output)
        confidence = float(output[class_id])
        label = LABELS[class_id]
        return label, confidence, output

    except Exception as e:
        print(f"❌ Error dalam klasifikasi: {e}")
        return "ERROR", 0.0, None

# ========================
# FUNGSI UPDATE FRAME
# ========================
def update_frame():
    # FIX 2: prev_time dikelola di dalam update_frame dengan global
    global prev_time

    ret, frame = cap.read()

    if not ret:
        lbl_status.config(text="❌ Kamera Error", fg="red")
        root.after(100, update_frame)
        return

    # FIX 3: Hitung FPS di dalam update_frame
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label, confidence, output = classify_image(roi)

        if label != "ERROR":
            prediction_history.append(label)
            # FIX 4: stable_label didefinisikan dengan benar di dalam blok ini
            stable_label = Counter(prediction_history).most_common(1)[0][0]
            color = CLASS_COLORS.get(stable_label, "white")

            # FIX 5: cv2.putText untuk label dipindah ke dalam update_frame
            cv2.putText(
                frame,
                f"{stable_label} ({confidence * 100:.1f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if confidence >= CONF_THRESHOLD:
                lbl_class.config(text=stable_label.upper(), fg=color)
            else:
                lbl_class.config(text="TIDAK YAKIN", fg="yellow")

            lbl_conf.config(text=f"Confidence: {confidence * 100:.2f}%")

            if confidence >= SAVE_THRESHOLD:
                filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(
                    f"hasil_klasifikasi/{stable_label}_{filename}.jpg",
                    frame
                )

            # FIX 6: Cek output tidak None sebelum di-loop
            if output is not None:
                print("\n======================")
                for i, score in enumerate(output):
                    print(f"{LABELS[i]:10s}: {score * 100:.2f}%")
                print("======================")

    # FIX 7: cv2.putText untuk FPS dipindah ke dalam update_frame
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

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
update_frame()
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Aplikasi ditutup")