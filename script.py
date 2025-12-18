import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound
import time
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6

AUDIO_MAP = {
    0: "assets/selamat.mp3",
    1: "assets/berjuang.mp3",
    2: "assets/sukses.mp3",
    3: "assets/clap.mp3",
}

model = tf.keras.models.load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)

last_played = None
cooldown = 2
last_time = 0

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    label = f"Class {class_id} ({confidence:.2f})"

    current_time = time.time()
    if confidence > CONFIDENCE_THRESHOLD:
        if class_id != last_played or (current_time - last_time) > cooldown:
            audio_file = AUDIO_MAP.get(class_id)
            if audio_file:
                playsound(audio_file, block=False)
                last_played = class_id
                last_time = current_time

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Selamat, Berjuang, Sukses", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
