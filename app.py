import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load model safely
MODEL_PATH = "asl_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Define labels (update these as per your model training)
labels = [chr(i) for i in range(65, 91)]  # A-Z

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("ASL Hand Gesture Recognition")
st.write("Show a hand gesture to your webcam and see the predicted letter!")

# Video transformer
class ASLVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)
        h, w, _ = img.shape

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                if len(landmarks) == model.input_shape[1]:
                    prediction = model.predict(np.array([landmarks]))[0]
                    pred_label = labels[np.argmax(prediction)]
                    cv2.putText(img, f'Predicted: {pred_label}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

# WebRTC Stream
webrtc_streamer(key="asl-stream", video_transformer_factory=ASLVideoTransformer)
