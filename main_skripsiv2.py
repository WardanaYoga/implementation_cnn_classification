import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("modelv5.keras")

# Nama kelas (sesuaikan urutan saat training)
class_names = [
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic"
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize untuk model
    img = cv2.resize(frame, (224, 224))

    img = img.astype(np.float32)

    # Preprocessing MobileNetV2
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    # Prediksi
    prediction = model.predict(img, verbose=0)

    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    label = f"{class_names[class_id]} ({confidence*100:.2f}%)"

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Klasifikasi Sampah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()