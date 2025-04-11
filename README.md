# 🖐️ Hand Tracking & 🧠 Image Recognition Script

This Python script allows you to run **real-time hand tracking using your webcam** or perform **image classification** using a pre-trained MobileNetV2 model. It combines computer vision techniques with deep learning to demonstrate two useful applications.

## 📦 Requirements

Make sure you have the following libraries installed:

```bash
pip install opencv-python mediapipe tensorflow numpy
```

## 📁 File Structure

```plaintext
hand_tracking.py         # Main script file
myimage.png              # Image to be used for classification (optional)
```

## 🚀 How to Use

Run the script:

```bash
python hand_tracking.py
```

You will be prompted to choose a mode:

```
Choose a mode:
1 - Hand Tracking (Webcam)
2 - Image Recognition (Static Image)
```

### Mode 1: Hand Tracking

- Activates your webcam.
- Uses [MediaPipe](https://google.github.io/mediapipe/) to detect and track hands in real-time.
- Displays hand landmarks drawn on the live video feed.
- Press `q` to quit.

### Mode 2: Image Recognition

- Loads a static image file (`myimage.png`).
- Uses the [MobileNetV2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) model pretrained on ImageNet.
- Prints the top 3 predictions with confidence scores.
- Replace `'myimage.png'` in the script with your own image if needed.

## 📌 Functions Explained

### `handtrack()`
- Initializes the webcam and MediaPipe Hands module.
- Converts each frame to RGB and processes it to find hand landmarks.
- Draws the landmarks on the video stream.
- Displays the live stream with annotations.

### `imgrecognition()`
- Loads the MobileNetV2 model with ImageNet weights.
- Loads and preprocesses the image for classification.
- Prints the top 3 predictions.

## ⚠️ Notes

- Ensure your webcam is connected and accessible.
- Place your image file (`myimage.png`) in the same directory or provide the correct path.
- You can enhance the script with `argparse` for better control from the command line.

## 📄 License

This script is provided for educational and experimental purposes.

