import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ---------------------- Hand Tracking ---------------------- #
def handtrack():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------- Image Recognition ---------------------- #
def imgrecognition():
    model = MobileNetV2(weights='imagenet')

    img_path = 'myimage.png'  # Replace with your image filename
    if not os.path.exists(img_path):
        print(f"[ERROR] Image file '{img_path}' not found.")
        return

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]

    print("Top Predictions:")
    for i, (imagenetID, label, prob) in enumerate(decoded):
        print(f"{i+1}. {label}: {prob*100:.2f}%")


# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    print("Choose a mode:")
    print("1 - Hand Tracking (Webcam)")
    print("2 - Image Recognition (Static Image)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        handtrack()
    elif choice == '2':
        imgrecognition()
    else:
        print("Invalid input. Please enter 1 or 2.")
