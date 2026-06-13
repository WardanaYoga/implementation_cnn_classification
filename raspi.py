"""
=================================================
SISTEM KLASIFIKASI SAMPAH - Raspberry Pi Version
=================================================
Cara menjalankan:
  - Mode GUI (dengan monitor)   : python3 waste_classifier_raspi.py
  - Mode Headless (tanpa monitor): python3 waste_classifier_raspi.py --headless

Instalasi dependensi di Raspi:
  pip install tflite-runtime opencv-python-headless numpy pillow

Jika pakai full TensorFlow (bukan tflite-runtime):
  Ganti bagian import interpreter di bawah (sudah diberi komentar)
=================================================
"""

import cv2
import numpy as np
import sys
import os
import time
import csv
import argparse
from collections import deque, Counter
from datetime import datetime

# ========================
# ARGUMEN CLI
# ========================
parser = argparse.ArgumentParser(description="Sistem Klasifikasi Sampah - Raspi")
parser.add_argument("--headless", action="store_true",
                    help="Jalankan tanpa GUI (cocok untuk SSH / tanpa monitor)")
parser.add_argument("--threads", type=int, default=4,
                    help="Jumlah thread CPU untuk inferensi (default: 4)")
parser.add_argument("--width", type=int, default=320,
                    help="Lebar resolusi kamera (default: 320)")
parser.add_argument("--height", type=int, default=240,
                    help="Tinggi resolusi kamera (default: 240)")
parser.add_argument("--camera", type=int, default=0,
                    help="Index kamera (default: 0)")
args = parser.parse_args()

HEADLESS = args.headless

# ========================
# IMPORT GUI (opsional)
# ========================
if not HEADLESS:
    try:
        import tkinter as tk
        from tkinter import Label, Button, Frame
        from PIL import Image, ImageTk
        GUI_AVAILABLE = True
    except ImportError:
        print("⚠️  Tkinter tidak tersedia, beralih ke mode headless")
        HEADLESS = True
        GUI_AVAILABLE = False
else:
    GUI_AVAILABLE = False

# ========================
# IMPORT TFLITE
# ========================
# Coba tflite-runtime dulu (ringan, cocok Raspi)
# Jika tidak ada, fallback ke full TensorFlow
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("✅ Menggunakan tflite-runtime (optimal untuk Raspi)")
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("⚠️  tflite-runtime tidak ditemukan, menggunakan TensorFlow penuh")
        print("    Disarankan: pip install tflite-runtime")
    except ImportError:
        print("❌ Tidak ada TFLite atau TensorFlow yang terinstall!")
        print("   Jalankan: pip install tflite-runtime")
        sys.exit(1)

# ========================
# KONFIGURASI
# ========================
MODEL_PATH            = "mobilenetv2model.tflite"
IMG_SIZE              = 224
LABELS                = ["glass", "metal", "organic", "paper", "plastic"]

CONF_THRESHOLD        = 0.85
ENTROPY_THRESHOLD     = 0.70
CONSISTENCY_THRESHOLD = 0.70
MIN_HISTORY_FRAMES    = 5
STD_DEV_THRESHOLD     = 20
COUNTDOWN_SECONDS     = 3

# Resolusi dari argumen CLI
CAM_WIDTH  = args.width
CAM_HEIGHT = args.height
NUM_THREADS = args.threads

# ========================
# INISIALISASI DIREKTORI & CSV
# ========================
os.makedirs("hasil_klasifikasi", exist_ok=True)
CSV_PATH = "hasil_klasifikasi/log_deteksi.csv"

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["No", "Waktu", "Label", "Confidence", "Entropy", "File"])

with open(CSV_PATH, "r") as f:
    detection_count = sum(1 for _ in f) - 1

# ========================
# LOAD MODEL TFLITE
# ========================
try:
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        num_threads=NUM_THREADS  # Manfaatkan semua core Raspi 4
    )
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model dimuat | Threads: {NUM_THREADS}")
    print(f"   Input  shape : {input_details[0]['shape']}")
    print(f"   Output shape : {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    sys.exit(1)

# ========================
# INISIALISASI KAMERA
# ========================
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"❌ Kamera index {args.camera} tidak bisa dibuka")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Kurangi buffer lag di Raspi

print(f"✅ Kamera aktif | Resolusi: {CAM_WIDTH}x{CAM_HEIGHT}")

# ========================
# STATE & VARIABEL GLOBAL
# ========================
# IDLE → COUNTDOWN → DETECTING → DONE → (ULANGI) → IDLE
STATE            = "IDLE"
countdown_start  = None
prediction_history = deque(maxlen=10)
prev_time        = time.time()
last_saved_label = None
last_saved_conf  = None
last_saved_file  = None

# ========================
# WARNA
# ========================
CLASS_COLORS_BGR = {
    "glass":   (255, 255, 0),
    "metal":   (200, 200, 200),
    "organic": (0, 255, 0),
    "paper":   (0, 165, 255),
    "plastic": (0, 255, 255),
}
CLASS_COLORS_TK = {
    "glass":   "cyan",
    "metal":   "lightgray",
    "organic": "lime",
    "paper":   "orange",
    "plastic": "yellow",
}

# ========================
# FUNGSI HELPER
# ========================
def preprocess_image(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = (img / 127.5) - 1.0          # MobileNetV2 preprocess manual
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(roi):
    try:
        img = preprocess_image(roi)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output     = interpreter.get_tensor(output_details[0]['index'])[0]
        if len(output) != len(LABELS):
            return "ERROR", 0.0, None
        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])
        label      = LABELS[class_id]
        return label, confidence, output
    except Exception as e:
        print(f"❌ Error klasifikasi: {e}")
        return "ERROR", 0.0, None

def has_object_in_roi(roi):
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    return std_dev > STD_DEV_THRESHOLD, std_dev

def compute_entropy(output):
    probs       = np.clip(np.array(output, dtype=np.float64), 1e-9, 1.0)
    probs      /= probs.sum()
    entropy     = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(LABELS)))

def get_stable_prediction(history, current_conf):
    if len(history) < MIN_HISTORY_FRAMES:
        return None, 0.0
    most_common, count = Counter(history).most_common(1)[0]
    consistency = count / len(history)
    if consistency >= CONSISTENCY_THRESHOLD and current_conf >= CONF_THRESHOLD:
        return most_common, consistency
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
    print(f"💾 [{detection_count}] Tersimpan: {filename} | {label} {confidence*100:.1f}%")
    return filename

def print_status(msg):
    """Print status ke terminal dengan timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ========================
# LOGIKA DETEKSI (shared)
# Dipakai oleh mode GUI maupun headless
# ========================
def process_frame(frame):
    """
    Proses satu frame sesuai STATE saat ini.
    Return: frame yang sudah diberi anotasi, dict info
    """
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file

    h, w = frame.shape[:2]
    x1 = int(w * 0.25)
    y1 = int(h * 0.20)
    x2 = int(w * 0.75)
    y2 = int(h * 0.80)
    roi = frame[y1:y2, x1:x2]

    info = {
        "label":       "---",
        "confidence":  0.0,
        "entropy":     0.0,
        "consistency": 0.0,
        "status":      "",
        "state":       STATE,
        "saved_file":  None,
    }

    # ---- IDLE ----
    if STATE == "IDLE":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(frame, "Siapkan sampah di sini",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (128, 128, 128), 1)
        info["status"] = "Siapkan sampah, lalu tekan MULAI / Enter"

    # ---- COUNTDOWN ----
    elif STATE == "COUNTDOWN":
        elapsed   = time.time() - countdown_start
        remaining = COUNTDOWN_SECONDS - int(elapsed)
        if remaining <= 0:
            STATE = "DETECTING"
            info["status"] = "Mendeteksi..."
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Mulai dalam {remaining}...",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.putText(frame, str(remaining),
                        (w // 2 - 20, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6)
            info["status"] = f"Mulai dalam {remaining} detik..."

    # ---- DETECTING ----
    elif STATE == "DETECTING":
        object_found, std_dev = has_object_in_roi(roi)

        if not object_found:
            prediction_history.clear()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(frame, "Tidak ada objek",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (128, 128, 128), 1)
            info["status"] = f"Objek tidak terdeteksi (std={std_dev:.1f})"
            info["label"]  = "TIDAK ADA OBJEK"
        else:
            label, confidence, output = classify_image(roi)

            if label != "ERROR" and output is not None:
                entropy = compute_entropy(output)
                prediction_history.append(label)
                stable_label, consistency = get_stable_prediction(
                    prediction_history, confidence)

                info["label"]       = label
                info["confidence"]  = confidence
                info["entropy"]     = entropy
                info["consistency"] = consistency

                if entropy > ENTROPY_THRESHOLD:
                    prediction_history.clear()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "Tidak Yakin",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 165, 255), 2)
                    info["status"] = "Model ragu (entropy tinggi)"
                    info["label"]  = "TIDAK YAKIN"

                elif stable_label is None:
                    progress = len(prediction_history)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Mendeteksi... ({label})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 255), 2)
                    # Progress bar
                    bar_w = int((x2 - x1) * progress / MIN_HISTORY_FRAMES)
                    cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 12), (50, 50, 50), -1)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 12),
                                  (0, 255, 255), -1)
                    info["status"] = f"Mengumpulkan ({progress}/{MIN_HISTORY_FRAMES})"

                else:
                    # ✅ Stabil → simpan
                    box_color  = CLASS_COLORS_BGR.get(stable_label, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(frame,
                                f"{stable_label.upper()} {confidence*100:.1f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, box_color, 2)

                    saved_file = save_result(frame, stable_label, confidence, entropy)

                    last_saved_label = stable_label
                    last_saved_conf  = confidence
                    last_saved_file  = saved_file

                    info["label"]      = stable_label
                    info["saved_file"] = saved_file
                    info["status"]     = f"Tersimpan! Tekan ULANGI / R"
                    STATE = "DONE"

    # ---- DONE ----
    elif STATE == "DONE":
        if last_saved_label:
            box_color = CLASS_COLORS_BGR.get(last_saved_label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(frame, f"✓ {last_saved_label.upper()} - TERSIMPAN",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, box_color, 2)
        info["label"]  = last_saved_label or "---"
        info["status"] = "Tekan ULANGI / R untuk deteksi berikutnya"

    return frame, info


# ==================================================
# MODE GUI (Tkinter)
# ==================================================
def run_gui():
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file

    root = tk.Tk()
    root.title("Klasifikasi Sampah - Raspberry Pi")
    root.geometry("780x580")
    root.configure(bg="#1e1e1e")
    root.resizable(False, False)

    Label(root, text="SISTEM KLASIFIKASI SAMPAH",
          font=("Arial", 17, "bold"),
          fg="white", bg="#1e1e1e").pack(pady=8)

    main_frame = Frame(root, bg="#1e1e1e")
    main_frame.pack()

    # Kamera
    cam_frame = Frame(main_frame, bg="black", width=400, height=300,
                      relief=tk.SUNKEN, borderwidth=2)
    cam_frame.grid(row=0, column=0, padx=10, pady=8)
    cam_frame.pack_propagate(False)
    cam_label = Label(cam_frame, bg="black")
    cam_label.pack(fill=tk.BOTH, expand=True)

    # Info
    info_frame = Frame(main_frame, bg="#2b2b2b", width=320, height=300,
                       relief=tk.RAISED, borderwidth=2)
    info_frame.grid(row=0, column=1, padx=10, pady=8)
    info_frame.pack_propagate(False)

    Label(info_frame, text="HASIL DETEKSI",
          font=("Arial", 14, "bold"),
          fg="white", bg="#2b2b2b").pack(pady=8)

    res_frame = Frame(info_frame, bg="#363636", relief=tk.GROOVE, borderwidth=2)
    res_frame.pack(pady=4, padx=12, fill=tk.BOTH)

    lbl_class = Label(res_frame, text="---",
                      font=("Arial", 26, "bold"),
                      fg="gray", bg="#363636", height=2)
    lbl_class.pack(pady=8)

    lbl_conf        = Label(info_frame, text="Confidence: -",
                            font=("Arial", 11), fg="white",  bg="#2b2b2b")
    lbl_conf.pack(pady=2)
    lbl_entropy     = Label(info_frame, text="Entropy: -",
                            font=("Arial", 10), fg="lightgray", bg="#2b2b2b")
    lbl_entropy.pack(pady=1)
    lbl_consistency = Label(info_frame, text="Konsistensi: -",
                            font=("Arial", 10), fg="lightgray", bg="#2b2b2b")
    lbl_consistency.pack(pady=1)
    lbl_count       = Label(info_frame, text=f"Total: {detection_count}",
                            font=("Arial", 10), fg="lightblue", bg="#2b2b2b")
    lbl_count.pack(pady=3)
    lbl_status      = Label(info_frame,
                            text="⏳ Siapkan sampah, tekan MULAI",
                            font=("Arial", 10), fg="orange", bg="#2b2b2b",
                            wraplength=290, justify="center")
    lbl_status.pack(pady=5)

    # Tombol
    btn_frame = Frame(root, bg="#1e1e1e")
    btn_frame.pack(pady=8)

    def on_start():
        global STATE, countdown_start
        if STATE == "IDLE":
            STATE = "COUNTDOWN"
            countdown_start = time.time()
            prediction_history.clear()
            btn_start.config(state=tk.DISABLED, bg="#555")
            btn_reset.config(state=tk.NORMAL,   bg="orange")

    def on_reset():
        global STATE, last_saved_label, last_saved_conf, last_saved_file
        STATE = "IDLE"
        prediction_history.clear()
        last_saved_label = last_saved_conf = last_saved_file = None
        btn_start.config(state=tk.NORMAL, bg="green")
        btn_reset.config(state=tk.DISABLED, bg="#555")
        lbl_class.config(text="---", fg="gray")
        lbl_conf.config(text="Confidence: -")
        lbl_entropy.config(text="Entropy: -")
        lbl_consistency.config(text="Konsistensi: -")
        lbl_status.config(text="⏳ Siapkan sampah, tekan MULAI", fg="orange")

    def on_exit():
        cap.release()
        root.destroy()

    btn_start = Button(btn_frame, text="▶ MULAI",
                       font=("Arial", 11, "bold"), bg="green", fg="white",
                       width=12, height=2, command=on_start)
    btn_start.grid(row=0, column=0, padx=8)

    btn_reset = Button(btn_frame, text="🔄 ULANGI",
                       font=("Arial", 11, "bold"), bg="#555", fg="white",
                       width=12, height=2, state=tk.DISABLED, command=on_reset)
    btn_reset.grid(row=0, column=1, padx=8)

    btn_exit = Button(btn_frame, text="✖ KELUAR",
                      font=("Arial", 11, "bold"), bg="red", fg="white",
                      width=12, height=2, command=on_exit)
    btn_exit.grid(row=0, column=2, padx=8)

    # Bind keyboard shortcut
    root.bind("<Return>", lambda e: on_start())
    root.bind("<r>",      lambda e: on_reset())
    root.bind("<q>",      lambda e: on_exit())

    prev_state = [STATE]

    def update():
        global prev_time
        ret, frame = cap.read()
        if not ret:
            lbl_status.config(text="❌ Kamera Error", fg="red")
            root.after(200, update)
            return

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame, info = process_frame(frame)

        # Update FPS overlay
        cv2.putText(frame, f"FPS:{fps:.1f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Tampilkan ke Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_res = cv2.resize(frame_rgb, (400, 300))
        img_tk    = ImageTk.PhotoImage(Image.fromarray(frame_res))
        cam_label.imgtk = img_tk
        cam_label.configure(image=img_tk)

        # Update panel info
        lbl_status.config(text=info["status"],
                          fg="lime" if STATE == "DONE" else
                             "yellow" if STATE in ("COUNTDOWN", "DETECTING") else
                             "orange")

        if info["label"] not in ("---", "TIDAK ADA OBJEK", "TIDAK YAKIN"):
            tk_color = CLASS_COLORS_TK.get(info["label"].lower(), "white")
            lbl_class.config(text=info["label"].upper(), fg=tk_color)
        elif info["label"] in ("TIDAK ADA OBJEK", "TIDAK YAKIN"):
            lbl_class.config(text=info["label"], fg="gray")

        if info["confidence"] > 0:
            lbl_conf.config(text=f"Confidence: {info['confidence']*100:.2f}%")
        if info["entropy"] > 0:
            lbl_entropy.config(
                text=f"Entropy: {info['entropy']:.3f} "
                     f"{'⚠️' if info['entropy'] > ENTROPY_THRESHOLD else '✅'}")
        if info["consistency"] > 0:
            lbl_consistency.config(text=f"Konsistensi: {info['consistency']*100:.0f}%")

        lbl_count.config(text=f"Total: {detection_count}")

        # Auto-update tombol saat state berubah
        if STATE == "DONE" and prev_state[0] != "DONE":
            btn_start.config(state=tk.DISABLED, bg="#555")
            btn_reset.config(state=tk.NORMAL, bg="orange")
        prev_state[0] = STATE

        root.after(50, update)   # 50ms = ~20 FPS, hemat CPU Raspi

    print_status("GUI dimulai | Enter=Mulai | R=Ulangi | Q=Keluar")
    update()
    root.mainloop()


# ==================================================
# MODE HEADLESS (Terminal)
# ==================================================
def run_headless():
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file

    print("\n" + "="*50)
    print("  MODE HEADLESS - Kontrol via keyboard")
    print("  Enter / S = Mulai deteksi")
    print("  R         = Ulangi")
    print("  Q         = Keluar")
    print("="*50 + "\n")

    # Non-blocking keyboard input di Linux
    import threading

    def keyboard_listener():
        global STATE, countdown_start, prediction_history
        global last_saved_label, last_saved_conf, last_saved_file
        while True:
            try:
                key = input().strip().lower()
                if key in ("", "s") and STATE == "IDLE":
                    STATE = "COUNTDOWN"
                    countdown_start = time.time()
                    prediction_history.clear()
                    print_status("▶ Deteksi dimulai! Hitung mundur...")
                elif key == "r":
                    STATE = "IDLE"
                    prediction_history.clear()
                    last_saved_label = last_saved_conf = last_saved_file = None
                    print_status("🔄 Reset. Siapkan sampah berikutnya, tekan Enter")
                elif key == "q":
                    print_status("👋 Keluar...")
                    cap.release()
                    os._exit(0)
            except EOFError:
                break

    t = threading.Thread(target=keyboard_listener, daemon=True)
    t.start()

    print_status(f"STATE: {STATE} | Siapkan sampah lalu tekan Enter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print_status("❌ Kamera Error")
            time.sleep(0.5)
            continue

        prev_state = STATE
        frame, info = process_frame(frame)

        # Print status saat state berubah atau ada update penting
        if STATE != prev_state:
            print_status(f"STATE: {STATE} | {info['status']}")

        if STATE == "DETECTING" and info["label"] not in ("---", "TIDAK ADA OBJEK"):
            print(f"\r  [{info['label']:10s}] conf:{info['confidence']*100:.1f}%"
                  f" ent:{info['entropy']:.2f}"
                  f" hist:{len(prediction_history)}/{MIN_HISTORY_FRAMES}   ",
                  end="", flush=True)

        if STATE == "DONE" and prev_state == "DETECTING":
            print()  # newline setelah progress
            print_status(f"✅ Hasil: {last_saved_label} "
                         f"({last_saved_conf*100:.1f}%) → {last_saved_file}")
            print_status("Tekan R untuk ulangi, Q untuk keluar")

        # Simpan preview frame ke file untuk debugging (opsional)
        cv2.imwrite("preview_latest.jpg", frame)

        time.sleep(0.05)  # ~20 FPS


# ========================
# ENTRY POINT
# ========================
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Klasifikasi Sampah - Raspberry Pi")
    print(f"  Mode     : {'HEADLESS' if HEADLESS else 'GUI'}")
    print(f"  Resolusi : {CAM_WIDTH}x{CAM_HEIGHT}")
    print(f"  Threads  : {NUM_THREADS}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"{'='*50}\n")

    if HEADLESS:
        run_headless()
    else:
        run_gui()

    cap.release()
    print("✅ Selesai")
