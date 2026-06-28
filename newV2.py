"""
=================================================
BENCHMARK SISTEM KLASIFIKASI SAMPAH
=================================================
Mengukur dan mencatat:
  4.3 - Waktu inferensi (ms)
  4.4 - FPS kamera
  4.5 - Penggunaan CPU & memori
  4.6 - Transmisi MQTT (status, payload)
  4.7 - Delay end-to-end & analisis paket

Output CSV:
  hasil_benchmark/inferensi.csv
  hasil_benchmark/fps_cpu.csv
  hasil_benchmark/mqtt_delay.csv
  hasil_benchmark/ringkasan.csv

Cara menjalankan:
  python3 benchmark_sistem.py --n 30
  (default 30 sampel pengujian)
=================================================
"""

import cv2
import numpy as np
import time
import csv
import os
import psutil
import json
import argparse
import threading
from datetime import datetime
from collections import deque

# ── Import TFLite ─────────────────────────────────────
try:
    from tflite_runtime.interpreter import Interpreter
    print("✓ Menggunakan tflite-runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("✓ Menggunakan TensorFlow penuh")

# ── Import MQTT ───────────────────────────────────────
import paho.mqtt.client as mqtt

# ========================
# KONFIGURASI
# ========================
MODEL_PATH    = "/home/aicenter/smartwaste/venv/newfinalmobilenetv2waste.tflite"
LABELS        = ["glass", "metal", "organic", "paper", "plastic"]
IMG_SIZE      = 224
CONF_THRESHOLD = 0.60

# MQTT
MQTT_BROKER   = "xxxx.s1.eu.hivemq.cloud"   # ganti dengan hostname kamu
MQTT_PORT     = 8883
MQTT_TOPIC    = "waste/benchmark"
MQTT_USERNAME = "username_kamu"
MQTT_PASSWORD = "password_kamu"

# Output
OUTPUT_DIR    = "hasil_benchmark"
NUM_THREADS   = 4

# ========================
# ARGUMEN CLI
# ========================
parser = argparse.ArgumentParser()
parser.add_argument("--n",      type=int, default=30,
                    help="Jumlah sampel pengujian (default: 30)")
parser.add_argument("--camera", type=int, default=0,
                    help="Index kamera (default: 0)")
args = parser.parse_args()
N_SAMPLES = args.n

# ========================
# INISIALISASI
# ========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kamera
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ========================
# VARIABEL GLOBAL MQTT
# ========================
mqtt_client         = None
mqtt_receive_times  = {}   # {msg_id: timestamp_diterima}
mqtt_sent_times     = {}   # {msg_id: timestamp_dikirim}
mqtt_receive_lock   = threading.Lock()
mqtt_connected      = False
mqtt_sent_count     = 0
mqtt_received_count = 0

# ========================
# FUNGSI MQTT
# ========================
def on_connect(client, userdata, flags, reason_code, properties):
    global mqtt_connected
    if reason_code == 0:
        mqtt_connected = True
        client.subscribe(MQTT_TOPIC)
        print(f"✓ MQTT terhubung → {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"✗ MQTT gagal connect: {reason_code}")

def on_message(client, userdata, msg):
    global mqtt_received_count
    recv_time = time.time()
    try:
        data = json.loads(msg.payload.decode())
        msg_id = data.get("benchmark_id")
        with mqtt_receive_lock:
            mqtt_receive_times[msg_id] = recv_time
            mqtt_received_count += 1
    except Exception as e:
        print(f"✗ Error parse MQTT: {e}")

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        mqtt_client.tls_set()
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        time.sleep(2)  # tunggu koneksi
    except Exception as e:
        print(f"✗ MQTT tidak tersedia: {e}")
        mqtt_client = None

# ========================
# FUNGSI PREPROCESSING & INFERENSI
# ========================
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ((img.astype(np.float32) / 127.5) - 1.0)
    return np.expand_dims(img, axis=0)

def infer(frame):
    """Inferensi + ukur waktu. Return (label, confidence, waktu_ms)"""
    tensor = preprocess(frame)

    t_start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    t_end   = time.perf_counter()

    output     = interpreter.get_tensor(output_details[0]['index'])[0]
    idx        = int(np.argmax(output))
    confidence = float(output[idx])
    label      = LABELS[idx]
    waktu_ms   = (t_end - t_start) * 1000

    return label, confidence, waktu_ms

# ========================
# FUNGSI UKUR FPS & CPU
# ========================
def ukur_fps_cpu(n_frame=60):
    """
    Ukur FPS kamera dan CPU/RAM selama pengambilan n_frame frame.
    Return dict hasil pengukuran.
    """
    print(f"\n[4.4 & 4.5] Mengukur FPS dan CPU ({n_frame} frame)...")
    cpu_readings = []
    ram_readings = []
    frame_times  = []

    # Warmup
    for _ in range(5):
        cap.read()

    t_prev = time.perf_counter()
    for i in range(n_frame):
        ret, frame = cap.read()
        if not ret:
            continue
        t_now = time.perf_counter()
        frame_times.append(t_now - t_prev)
        t_prev = t_now

        # Jalankan inferensi juga supaya CPU load realistis
        infer(frame)

        cpu_readings.append(psutil.cpu_percent(interval=None))
        ram_readings.append(psutil.virtual_memory().percent)

    fps_list = [1.0 / ft for ft in frame_times if ft > 0]

    return {
        "fps_rata"   : round(float(np.mean(fps_list)), 2),
        "fps_min"    : round(float(np.min(fps_list)),  2),
        "fps_max"    : round(float(np.max(fps_list)),  2),
        "fps_std"    : round(float(np.std(fps_list)),  2),
        "cpu_rata"   : round(float(np.mean(cpu_readings)), 2),
        "cpu_max"    : round(float(np.max(cpu_readings)),  2),
        "ram_rata"   : round(float(np.mean(ram_readings)), 2),
        "ram_max"    : round(float(np.max(ram_readings)),  2),
    }

# ========================
# MAIN BENCHMARK
# ========================
def main():
    global mqtt_sent_count

    print("=" * 55)
    print("  BENCHMARK SISTEM KLASIFIKASI SAMPAH")
    print(f"  Jumlah sampel : {N_SAMPLES}")
    print(f"  Model         : {MODEL_PATH}")
    print("=" * 55)

    # Init MQTT
    init_mqtt()
    if not mqtt_connected:
        print("⚠ MQTT tidak terhubung — pengukuran delay dilewati")

    # ── WARMUP inferensi ──────────────────────────────
    print("\n[Warmup] 5 frame pertama dibuang...")
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            infer(frame)

    # ── 4.3 & 4.6 & 4.7 : Inferensi + MQTT ──────────
    print(f"\n[4.3 / 4.6 / 4.7] Mulai pengujian {N_SAMPLES} sampel...\n")
    print(f"{'No':<5} {'Label':<12} {'Conf':>7} {'Infer(ms)':>12} {'MQTT':>6}")
    print("-" * 45)

    rows_inferensi = []

    for i in range(1, N_SAMPLES + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"  ✗ Frame {i} gagal dibaca, skip")
            continue

        label, conf, infer_ms = infer(frame)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Kirim via MQTT + catat waktu kirim
        mqtt_status = "skip"
        if mqtt_client and mqtt_connected:
            payload = {
                "benchmark_id" : i,
                "timestamp"    : timestamp,
                "label"        : label,
                "confidence"   : round(conf * 100, 2),
                "infer_ms"     : round(infer_ms, 3),
            }
            send_time = time.time()
            mqtt_sent_times[i] = send_time
            result = mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
            mqtt_status  = "OK" if result.rc == 0 else "FAIL"
            mqtt_sent_count += 1

        print(f"  {i:<4} {label:<12} {conf*100:>6.1f}%  {infer_ms:>9.2f}ms  {mqtt_status:>6}")

        rows_inferensi.append({
            "No"            : i,
            "Timestamp"     : timestamp,
            "Label"         : label,
            "Confidence (%)" : round(conf * 100, 2),
            "Waktu Inferensi (ms)" : round(infer_ms, 3),
            "MQTT Status"   : mqtt_status,
        })

        time.sleep(0.5)  # jeda antar sampel

    # Tunggu semua pesan MQTT diterima (max 5 detik)
    if mqtt_connected:
        print("\n  Menunggu konfirmasi MQTT...")
        time.sleep(5)

    # ── 4.4 & 4.5 : FPS + CPU ────────────────────────
    hasil_fps_cpu = ukur_fps_cpu(n_frame=60)

    # ── Hitung delay MQTT ────────────────────────────
    rows_mqtt = []
    delays    = []

    for msg_id, send_t in mqtt_sent_times.items():
        recv_t = mqtt_receive_times.get(msg_id)
        if recv_t:
            delay_ms = (recv_t - send_t) * 1000
            delays.append(delay_ms)
            rows_mqtt.append({
                "No"              : msg_id,
                "Waktu Kirim"     : datetime.fromtimestamp(send_t).strftime("%H:%M:%S.%f")[:-3],
                "Waktu Terima"    : datetime.fromtimestamp(recv_t).strftime("%H:%M:%S.%f")[:-3],
                "Delay (ms)"      : round(delay_ms, 2),
                "Status"          : "Diterima",
            })
        else:
            rows_mqtt.append({
                "No"              : msg_id,
                "Waktu Kirim"     : datetime.fromtimestamp(send_t).strftime("%H:%M:%S.%f")[:-3],
                "Waktu Terima"    : "-",
                "Delay (ms)"      : "-",
                "Status"          : "Tidak Diterima",
            })

    # ── Hitung statistik inferensi ────────────────────
    infer_times = [r["Waktu Inferensi (ms)"] for r in rows_inferensi]
    conf_values = [r["Confidence (%)"] for r in rows_inferensi]

    # ── Ringkasan ─────────────────────────────────────
    packet_loss = ((mqtt_sent_count - mqtt_received_count) / mqtt_sent_count * 100
                   if mqtt_sent_count > 0 else 0)

    ringkasan = {
        # 4.3 Inferensi
        "Rata-rata Waktu Inferensi (ms)" : round(float(np.mean(infer_times)), 3),
        "Min Waktu Inferensi (ms)"       : round(float(np.min(infer_times)),  3),
        "Maks Waktu Inferensi (ms)"      : round(float(np.max(infer_times)),  3),
        "Std Dev Inferensi (ms)"         : round(float(np.std(infer_times)),  3),
        "Rata-rata Confidence (%)"       : round(float(np.mean(conf_values)), 2),
        # 4.4 FPS
        "Rata-rata FPS"                  : hasil_fps_cpu["fps_rata"],
        "Min FPS"                        : hasil_fps_cpu["fps_min"],
        "Maks FPS"                       : hasil_fps_cpu["fps_max"],
        # 4.5 CPU & RAM
        "Rata-rata CPU (%)"              : hasil_fps_cpu["cpu_rata"],
        "Maks CPU (%)"                   : hasil_fps_cpu["cpu_max"],
        "Rata-rata RAM (%)"              : hasil_fps_cpu["ram_rata"],
        "Maks RAM (%)"                   : hasil_fps_cpu["ram_max"],
        # 4.6 & 4.7 MQTT
        "Paket Terkirim"                 : mqtt_sent_count,
        "Paket Diterima"                 : mqtt_received_count,
        "Packet Loss (%)"                : round(packet_loss, 2),
        "Rata-rata Delay MQTT (ms)"      : round(float(np.mean(delays)), 2) if delays else "-",
        "Min Delay MQTT (ms)"            : round(float(np.min(delays)),  2) if delays else "-",
        "Maks Delay MQTT (ms)"           : round(float(np.max(delays)),  2) if delays else "-",
    }

    # ========================
    # SIMPAN CSV
    # ========================
    # 1. CSV Inferensi
    path_inferensi = os.path.join(OUTPUT_DIR, "inferensi.csv")
    with open(path_inferensi, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_inferensi[0].keys())
        writer.writeheader()
        writer.writerows(rows_inferensi)

    # 2. CSV FPS & CPU
    path_fps = os.path.join(OUTPUT_DIR, "fps_cpu.csv")
    with open(path_fps, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Nilai"])
        for k, v in hasil_fps_cpu.items():
            writer.writerow([k, v])

    # 3. CSV MQTT Delay
    if rows_mqtt:
        path_mqtt = os.path.join(OUTPUT_DIR, "mqtt_delay.csv")
        with open(path_mqtt, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows_mqtt[0].keys())
            writer.writeheader()
            writer.writerows(rows_mqtt)

    # 4. CSV Ringkasan
    path_ringkasan = os.path.join(OUTPUT_DIR, "ringkasan.csv")
    with open(path_ringkasan, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Nilai"])
        for k, v in ringkasan.items():
            writer.writerow([k, v])

    # ========================
    # PRINT RINGKASAN
    # ========================
    print("\n" + "=" * 55)
    print("  RINGKASAN HASIL BENCHMARK")
    print("=" * 55)
    print(f"\n  [4.3] Waktu Inferensi")
    print(f"    Rata-rata : {ringkasan['Rata-rata Waktu Inferensi (ms)']} ms")
    print(f"    Min–Maks  : {ringkasan['Min Waktu Inferensi (ms)']} – {ringkasan['Maks Waktu Inferensi (ms)']} ms")
    print(f"    Std Dev   : {ringkasan['Std Dev Inferensi (ms)']} ms")

    print(f"\n  [4.4] FPS Kamera")
    print(f"    Rata-rata : {ringkasan['Rata-rata FPS']} FPS")
    print(f"    Min–Maks  : {ringkasan['Min FPS']} – {ringkasan['Maks FPS']} FPS")

    print(f"\n  [4.5] CPU & Memori")
    print(f"    CPU rata  : {ringkasan['Rata-rata CPU (%)']}%  (maks {ringkasan['Maks CPU (%)']}%)")
    print(f"    RAM rata  : {ringkasan['Rata-rata RAM (%)']}%  (maks {ringkasan['Maks RAM (%)']}%)")

    print(f"\n  [4.6] Transmisi MQTT")
    print(f"    Terkirim  : {ringkasan['Paket Terkirim']} paket")
    print(f"    Diterima  : {ringkasan['Paket Diterima']} paket")
    print(f"    Loss      : {ringkasan['Packet Loss (%)']}%")

    print(f"\n  [4.7] Delay End-to-End")
    print(f"    Rata-rata : {ringkasan['Rata-rata Delay MQTT (ms)']} ms")
    print(f"    Min–Maks  : {ringkasan['Min Delay MQTT (ms)']} – {ringkasan['Maks Delay MQTT (ms)']} ms")

    print(f"\n  File tersimpan di folder: {OUTPUT_DIR}/")
    print(f"    ✓ inferensi.csv")
    print(f"    ✓ fps_cpu.csv")
    print(f"    ✓ mqtt_delay.csv")
    print(f"    ✓ ringkasan.csv")
    print("=" * 55)

    cap.release()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()
