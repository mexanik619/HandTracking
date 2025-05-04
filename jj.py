# collect_asl_data.py
import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

DATA_DIR = "asl_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

label = input("Enter the label (A-Z): ").upper()
assert label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "Invalid label"

cap = cv2.VideoCapture(0)
sample_count = 0
max_samples = 100

while sample_count < max_samples:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            np.save(os.path.join(DATA_DIR, f"{label}_{sample_count}.npy"), np.array(landmarks))
            sample_count += 1
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Collecting {label}: {sample_count}/{max_samples}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
