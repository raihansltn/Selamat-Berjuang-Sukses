import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound
import time
import mediapipe as mp
import pygame
from collections import deque
MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6
PRED_BUFFER = 7

AUDIO_MAP = {
    0: "assets/selamat.mp3",
    1: "assets/berjuang.mp3",
    2: "assets/sukses.mp3",
    3: "assets/clap.mp3",
}

model = tf.keras.models.load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)

last_played = None
COOLDOWN = 2
last_time = 0

print("Press 'q' to quit")

pygame.mixer.init()

def play_audio(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)

pred_queue = deque(maxlen=PRED_BUFFER)
last_played = None
last_time = 0

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]

        x1, y1 = int(min(xs) * w), int(min(ys) * h)
        x2, y2 = int(max(xs) * w), int(max(ys) * h)

        pad = 20
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            img = cv2.resize(roi, IMG_SIZE)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img, verbose=0)
            cls = np.argmax(preds)
            conf = preds[0][cls]

            pred_queue.append(cls)

            stable_cls = max(set(pred_queue), key=pred_queue.count)

            label = f"Class {stable_cls} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            now = time.time()
            if conf > CONFIDENCE_THRESHOLD:
                if stable_cls != last_played or (now - last_time) > COOLDOWN:
                    play_audio(AUDIO_MAP[stable_cls])
                    last_played = stable_cls
                    last_time = now

    else:
        pred_queue.clear()
        cv2.putText(frame, "Tangannya mana liatin dulu, tangannya.", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()