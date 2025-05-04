import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Load your trained ASL model
MODEL_PATH = 'asl_model.h5'  # Replace with your ASL model path
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] ASL model file '{MODEL_PATH}' not found.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels (assumes 26 letters for ASL A-Z)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ---------------------- ASL Detection ---------------------- #
def asl_detection():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks as a flat array
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Convert to numpy and predict
                input_data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                class_id = np.argmax(prediction)
                confidence = prediction[class_id]

                # Display the predicted ASL letter
                label = f"{labels[class_id]} ({confidence*100:.1f}%)"
                cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)

        cv2.imshow("ASL Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    print("Starting ASL Gesture Detection...")
    asl_detection()
