"""
=================================================
SISTEM KLASIFIKASI SAMPAH - Raspberry Pi Version
=================================================
Cara menjalankan:
  - Mode GUI (dengan monitor)   : python3 waste_classifier_raspi.py
  - Mode Headless (tanpa monitor): python3 waste_classifier_raspi.py --headless

Instalasi dependensi di Raspi:
  pip install tflite-runtime opencv-python-headless numpy pillow psutil

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
import paho.mqtt.client as mqtt
import json
import requests
import threading
import psutil

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
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("✅ Menggunakan tflite-runtime (optimal untuk Raspi)")
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("⚠️  tflite-runtime tidak ditemukan, menggunakan TensorFlow penuh")
    except ImportError:
        print("❌ Tidak ada TFLite atau TensorFlow yang terinstall!")
        sys.exit(1)

# ========================
# KONFIGURASI
# ========================
MODEL_PATH            = "/home/aicenter/smartwaste/venv/newfinalmobilenetv2waste.tflite"
IMG_SIZE              = 224
LABELS                = ["glass", "metal", "organic", "paper", "plastic"]

CONF_THRESHOLD        = 0.85
ENTROPY_THRESHOLD     = 0.70
CONSISTENCY_THRESHOLD = 0.70
MIN_HISTORY_FRAMES    = 5
STD_DEV_THRESHOLD     = 20
COUNTDOWN_SECONDS     = 3

CAM_WIDTH   = args.width
CAM_HEIGHT  = args.height
NUM_THREADS = args.threads

# ========================
# MQTT
# ========================
MQTT_BROKER   = "327fdad9055149769b3bbe55f6ee8822.s1.eu.hivemq.cloud"
MQTT_PORT     = 8883
MQTT_TOPIC    = "waste/result"
MQTT_USERNAME = "klasifikasisampah"
MQTT_PASSWORD = "Pakraden.2026"

mqtt_client        = None
mqtt_sent_times    = {}   # {no_deteksi: timestamp_kirim}
mqtt_receive_times = {}   # {no_deteksi: timestamp_terima}
mqtt_receive_lock  = threading.Lock()
mqtt_sent_count    = 0
mqtt_received_count = 0

def init_mqtt():
    global mqtt_client

    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        mqtt_client.tls_set()

        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code == 0:
                print(f"✅ MQTT Connected → {MQTT_BROKER}:{MQTT_PORT}")
                # Subscribe ke topic sendiri untuk ukur delay
                client.subscribe(MQTT_TOPIC)
            else:
                print(f"❌ MQTT Gagal connect, kode: {reason_code}")

        def on_message(client, userdata, msg):
            global mqtt_received_count
            recv_time = time.time()
            try:
                data   = json.loads(msg.payload.decode())
                msg_no = data.get("no")
                with mqtt_receive_lock:
                    mqtt_receive_times[msg_no] = recv_time
                    mqtt_received_count += 1
            except Exception:
                pass

        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()

    except Exception as e:
        print(f"❌ MQTT tidak tersedia: {e}")
        mqtt_client = None

# ========================
# INISIALISASI DIREKTORI & CSV
# ========================
os.makedirs("hasil_klasifikasi", exist_ok=True)
os.makedirs("hasil_benchmark",   exist_ok=True)

CSV_PATH          = "hasil_klasifikasi/log_deteksi.csv"
CSV_INFERENSI     = "hasil_benchmark/inferensi.csv"
CSV_FPS_CPU       = "hasil_benchmark/fps_cpu.csv"
CSV_MQTT_DELAY    = "hasil_benchmark/mqtt_delay.csv"
CSV_RINGKASAN     = "hasil_benchmark/ringkasan.csv"

# Log deteksi utama
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["No", "Waktu", "Label", "Confidence", "Entropy", "File"])

with open(CSV_PATH, "r") as f:
    detection_count = sum(1 for _ in f) - 1

# CSV inferensi — header
if not os.path.exists(CSV_INFERENSI):
    with open(CSV_INFERENSI, "w", newline="") as f:
        csv.writer(f).writerow([
            "No", "Timestamp", "Label", "Confidence (%)",
            "Waktu Inferensi (ms)", "CPU (%)", "RAM (%)"
        ])

# ========================
# BUFFER PENGUKURAN
# ========================
buf_infer_ms  = []   # waktu inferensi tiap deteksi
buf_fps       = deque(maxlen=200)   # FPS frame loop
buf_cpu       = []   # CPU per deteksi
buf_ram       = []   # RAM per deteksi
prev_frame_t  = time.perf_counter()

# ========================
# LOAD MODEL TFLITE
# ========================
try:
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        num_threads=NUM_THREADS
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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print(f"✅ Kamera aktif | Resolusi: {CAM_WIDTH}x{CAM_HEIGHT}")

# ========================
# STATE & VARIABEL GLOBAL
# ========================
STATE              = "IDLE"
countdown_start    = None
prediction_history = deque(maxlen=10)
prev_time          = time.time()
last_saved_label   = None
last_saved_conf    = None
last_saved_file    = None

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
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(roi):
    """Inferensi + catat waktu, CPU, RAM."""
    try:
        img = preprocess_image(roi)

        # Ukur waktu inferensi
        t_start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        t_end   = time.perf_counter()

        output     = interpreter.get_tensor(output_details[0]['index'])[0]
        if len(output) != len(LABELS):
            return "ERROR", 0.0, None, 0.0

        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])
        label      = LABELS[class_id]
        infer_ms   = (t_end - t_start) * 1000

        return label, confidence, output, infer_ms

    except Exception as e:
        print(f"❌ Error klasifikasi: {e}")
        return "ERROR", 0.0, None, 0.0

def has_object_in_roi(roi):
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    return std_dev > STD_DEV_THRESHOLD, std_dev

def compute_entropy(output):
    probs   = np.clip(np.array(output, dtype=np.float64), 1e-9, 1.0)
    probs  /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(LABELS)))

def get_stable_prediction(history, current_conf):
    if len(history) < MIN_HISTORY_FRAMES:
        return None, 0.0
    most_common, count = Counter(history).most_common(1)[0]
    consistency = count / len(history)
    if consistency >= CONSISTENCY_THRESHOLD and current_conf >= CONF_THRESHOLD:
        return most_common, consistency
    return None, consistency

# ========================
# HTTP UPLOAD
# ========================
SERVER_URL = "http://192.168.0.102:5000/upload"

def upload_image_to_server(filepath, label, confidence, entropy, no):
    try:
        with open(filepath, "rb") as img_file:
            response = requests.post(
                SERVER_URL,
                files={"image": img_file},
                data={
                    "label"      : label,
                    "confidence" : round(confidence * 100, 2),
                    "entropy"    : round(entropy, 3),
                    "no"         : no,
                    "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                timeout=5
            )
        if response.status_code == 200:
            print(f"✅ Gambar terkirim ke server")
        else:
            print(f"❌ Upload gagal: {response.status_code}")
    except Exception as e:
        print(f"❌ Upload error: {e}")

# ========================
# SIMPAN HASIL + BENCHMARK
# ========================
def save_result(frame, label, confidence, entropy, infer_ms):
    global detection_count
    detection_count += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"hasil_klasifikasi/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    # Log deteksi utama
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            detection_count,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{confidence*100:.2f}%",
            f"{entropy:.3f}",
            filename
        ])

    # Catat CPU & RAM saat deteksi
    cpu_now = psutil.cpu_percent(interval=None)
    ram_now = psutil.virtual_memory().percent

    # Simpan ke buffer
    buf_infer_ms.append(infer_ms)
    buf_cpu.append(cpu_now)
    buf_ram.append(ram_now)

    # Tulis ke CSV inferensi
    with open(CSV_INFERENSI, "a", newline="") as f:
        csv.writer(f).writerow([
            detection_count,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            label,
            round(confidence * 100, 2),
            round(infer_ms, 3),
            cpu_now,
            ram_now
        ])

    print(f"💾 [{detection_count}] {filename} | {label} {confidence*100:.1f}% | {infer_ms:.1f}ms")
    upload_image_to_server(filename, label, confidence, entropy, detection_count)
    return filename

def send_to_server(label, confidence, entropy, filename):
    global mqtt_sent_count

    if mqtt_client is None:
        return

    payload = {
        "no"        : detection_count,
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label"     : label,
        "confidence": round(confidence * 100, 2),
        "entropy"   : round(entropy, 3),
        "filename"  : os.path.basename(filename)
    }

    try:
        send_time = time.time()
        mqtt_sent_times[detection_count] = send_time
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        mqtt_sent_count += 1
        print(f"✅ MQTT terkirim: {payload}")
    except Exception as e:
        print(f"❌ MQTT Publish Error: {e}")

# ========================
# SIMPAN RINGKASAN CSV
# (dipanggil saat program ditutup)
# ========================
def simpan_ringkasan():
    print("\n📊 Menyimpan hasil benchmark...")

    # ── FPS ──────────────────────────────────────────
    fps_list = list(buf_fps)

    # ── MQTT Delay ───────────────────────────────────
    delay_rows = []
    delays     = []

    for no, send_t in mqtt_sent_times.items():
        recv_t = mqtt_receive_times.get(no)
        if recv_t:
            delay_ms = (recv_t - send_t) * 1000
            delays.append(delay_ms)
            status = "Diterima"
        else:
            delay_ms = None
            status   = "Tidak Diterima"

        delay_rows.append({
            "No"           : no,
            "Waktu Kirim"  : datetime.fromtimestamp(send_t).strftime("%H:%M:%S.%f")[:-3],
            "Waktu Terima" : datetime.fromtimestamp(recv_t).strftime("%H:%M:%S.%f")[:-3] if recv_t else "-",
            "Delay (ms)"   : round(delay_ms, 2) if delay_ms else "-",
            "Status"       : status,
        })

    # Tulis CSV MQTT delay
    if delay_rows:
        with open(CSV_MQTT_DELAY, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=delay_rows[0].keys())
            writer.writeheader()
            writer.writerows(delay_rows)

    # ── FPS CPU CSV ───────────────────────────────────
    with open(CSV_FPS_CPU, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Nilai"])
        if fps_list:
            writer.writerow(["Rata-rata FPS",  round(float(np.mean(fps_list)), 2)])
            writer.writerow(["Min FPS",        round(float(np.min(fps_list)),  2)])
            writer.writerow(["Maks FPS",       round(float(np.max(fps_list)),  2)])
            writer.writerow(["Std Dev FPS",    round(float(np.std(fps_list)),  2)])
        if buf_cpu:
            writer.writerow(["Rata-rata CPU (%)", round(float(np.mean(buf_cpu)), 2)])
            writer.writerow(["Maks CPU (%)",      round(float(np.max(buf_cpu)),  2)])
        if buf_ram:
            writer.writerow(["Rata-rata RAM (%)", round(float(np.mean(buf_ram)), 2)])
            writer.writerow(["Maks RAM (%)",      round(float(np.max(buf_ram)),  2)])

    # ── Ringkasan ─────────────────────────────────────
    packet_loss = ((mqtt_sent_count - mqtt_received_count) / mqtt_sent_count * 100
                   if mqtt_sent_count > 0 else 0)

    with open(CSV_RINGKASAN, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Nilai"])

        # 4.3 Inferensi
        if buf_infer_ms:
            writer.writerow(["=== 4.3 Waktu Inferensi ===", ""])
            writer.writerow(["Rata-rata (ms)", round(float(np.mean(buf_infer_ms)), 3)])
            writer.writerow(["Min (ms)",       round(float(np.min(buf_infer_ms)),  3)])
            writer.writerow(["Maks (ms)",      round(float(np.max(buf_infer_ms)),  3)])
            writer.writerow(["Std Dev (ms)",   round(float(np.std(buf_infer_ms)),  3)])

        # 4.4 FPS
        if fps_list:
            writer.writerow(["=== 4.4 FPS Kamera ===", ""])
            writer.writerow(["Rata-rata FPS",  round(float(np.mean(fps_list)), 2)])
            writer.writerow(["Min FPS",        round(float(np.min(fps_list)),  2)])
            writer.writerow(["Maks FPS",       round(float(np.max(fps_list)),  2)])

        # 4.5 CPU & RAM
        if buf_cpu:
            writer.writerow(["=== 4.5 CPU & Memori ===", ""])
            writer.writerow(["Rata-rata CPU (%)", round(float(np.mean(buf_cpu)), 2)])
            writer.writerow(["Maks CPU (%)",      round(float(np.max(buf_cpu)),  2)])
            writer.writerow(["Rata-rata RAM (%)", round(float(np.mean(buf_ram)), 2)])
            writer.writerow(["Maks RAM (%)",      round(float(np.max(buf_ram)),  2)])

        # 4.6 & 4.7 MQTT
        writer.writerow(["=== 4.6 & 4.7 MQTT ===", ""])
        writer.writerow(["Paket Terkirim",    mqtt_sent_count])
        writer.writerow(["Paket Diterima",    mqtt_received_count])
        writer.writerow(["Packet Loss (%)",   round(packet_loss, 2)])
        if delays:
            writer.writerow(["Rata-rata Delay (ms)", round(float(np.mean(delays)), 2)])
            writer.writerow(["Min Delay (ms)",        round(float(np.min(delays)),  2)])
            writer.writerow(["Maks Delay (ms)",       round(float(np.max(delays)),  2)])
            writer.writerow(["Std Dev Delay (ms)",    round(float(np.std(delays)),  2)])

    print(f"✅ Tersimpan di folder hasil_benchmark/")
    print(f"   ├── inferensi.csv   ({len(buf_infer_ms)} deteksi)")
    print(f"   ├── fps_cpu.csv")
    print(f"   ├── mqtt_delay.csv  ({len(delay_rows)} paket)")
    print(f"   └── ringkasan.csv")

def print_status(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ========================
# LOGIKA DETEKSI (shared)
# ========================
def process_frame(frame):
    global STATE, countdown_start, prediction_history
    global last_saved_label, last_saved_conf, last_saved_file
    global prev_frame_t

    # ── Catat FPS frame loop ──
    now_t = time.perf_counter()
    fps_now = 1.0 / (now_t - prev_frame_t + 1e-9)
    buf_fps.append(fps_now)
    prev_frame_t = now_t

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
            # classify_image sekarang return 4 nilai (tambah infer_ms)
            label, confidence, output, infer_ms = classify_image(roi)

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
                    bar_w = int((x2 - x1) * progress / MIN_HISTORY_FRAMES)
                    cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 12), (50, 50, 50), -1)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 12),
                                  (0, 255, 255), -1)
                    info["status"] = f"Mengumpulkan ({progress}/{MIN_HISTORY_FRAMES})"

                else:
                    box_color  = CLASS_COLORS_BGR.get(stable_label, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(frame,
                                f"{stable_label.upper()} {confidence*100:.1f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, box_color, 2)

                    # save_result sekarang terima infer_ms
                    saved_file = save_result(frame, stable_label, confidence, entropy, infer_ms)
                    send_to_server(stable_label, confidence, entropy, saved_file)

                    last_saved_label = stable_label
                    last_saved_conf  = confidence
                    last_saved_file  = saved_file

                    info["label"]      = stable_label
                    info["saved_file"] = saved_file
                    info["status"]     = "Tersimpan! Tekan ULANGI / R"
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
    root.attributes("-fullscreen", True)
    root.configure(bg="#1e1e1e")
    root.resizable(False, False)

    Label(root, text="SISTEM KLASIFIKASI SAMPAH",
          font=("Arial", 17, "bold"),
          fg="white", bg="#1e1e1e").pack(pady=8)

    main_frame = Frame(root, bg="#1e1e1e")
    main_frame.pack()

    cam_frame = Frame(main_frame, bg="black", width=400, height=300,
                      relief=tk.SUNKEN, borderwidth=2)
    cam_frame.grid(row=0, column=0, padx=10, pady=8)
    cam_frame.pack_propagate(False)
    cam_label = Label(cam_frame, bg="black")
    cam_label.pack(fill=tk.BOTH, expand=True)

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

    # Tambahan: label benchmark live
    lbl_bench = Label(info_frame, text="Infer: - ms | FPS: -",
                      font=("Arial", 9), fg="#888", bg="#2b2b2b")
    lbl_bench.pack(pady=1)

    lbl_status = Label(info_frame,
                       text="⏳ Siapkan sampah, tekan MULAI",
                       font=("Arial", 10), fg="orange", bg="#2b2b2b",
                       wraplength=290, justify="center")
    lbl_status.pack(pady=5)

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
        simpan_ringkasan()
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

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame, info = process_frame(frame)

        cv2.putText(frame, f"FPS:{fps:.1f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_res = cv2.resize(frame_rgb, (400, 300))
        img_tk    = ImageTk.PhotoImage(Image.fromarray(frame_res))
        cam_label.imgtk = img_tk
        cam_label.configure(image=img_tk)

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

        # Update label benchmark live
        avg_infer = f"{np.mean(buf_infer_ms):.1f}" if buf_infer_ms else "-"
        avg_fps   = f"{np.mean(list(buf_fps)):.1f}" if buf_fps else "-"
        lbl_bench.config(text=f"Infer: {avg_infer} ms | FPS: {avg_fps}")

        if STATE == "DONE" and prev_state[0] != "DONE":
            btn_start.config(state=tk.DISABLED, bg="#555")
            btn_reset.config(state=tk.NORMAL, bg="orange")
        prev_state[0] = STATE

        root.after(50, update)

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
                    simpan_ringkasan()
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

        if STATE != prev_state:
            print_status(f"STATE: {STATE} | {info['status']}")

        if STATE == "DETECTING" and info["label"] not in ("---", "TIDAK ADA OBJEK"):
            avg_infer = f"{np.mean(buf_infer_ms):.1f}ms" if buf_infer_ms else "-"
            avg_fps   = f"{np.mean(list(buf_fps)):.1f}" if buf_fps else "-"
            print(f"\r  [{info['label']:10s}] conf:{info['confidence']*100:.1f}%"
                  f" ent:{info['entropy']:.2f}"
                  f" hist:{len(prediction_history)}/{MIN_HISTORY_FRAMES}"
                  f" | infer:{avg_infer} fps:{avg_fps}   ",
                  end="", flush=True)

        if STATE == "DONE" and prev_state == "DETECTING":
            print()
            print_status(f"✅ Hasil: {last_saved_label} "
                         f"({last_saved_conf*100:.1f}%) → {last_saved_file}")
            print_status("Tekan R untuk ulangi, Q untuk keluar + simpan benchmark")

        cv2.imwrite("preview_latest.jpg", frame)
        time.sleep(0.05)


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

    init_mqtt()

    try:
        if HEADLESS:
            run_headless()
        else:
            run_gui()
    finally:
        # Pastikan ringkasan selalu tersimpan meski program crash
        simpan_ringkasan()
        cap.release()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("✅ Selesai")
