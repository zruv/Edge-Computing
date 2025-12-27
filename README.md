# Edge Vision AI 👁️

A lightweight, high-performance object detection system designed for edge computing. It runs state-of-the-art **EfficientDet-Lite4** models on low-power devices (like Android phones via Termux) and serves a real-time, futuristic web dashboard to any device on the network.

## 🚀 Features

*   **Real-time Detection:** Uses TensorFlow Lite for efficient inference on edge hardware.
*   **Async Architecture:** Multi-threaded design decouples video capture, AI processing, and web streaming to maximize FPS and prevent lag.
*   **Web Dashboard:** Accessible via any browser with a responsive "Cyberpunk/Hacker" aesthetic.
*   **Edge Optimized:** Specifically tuned for running on Termux (Android) but compatible with standard Linux/macOS environments.
*   **Network Stream:** Consumes video from IP Webcam sources or local USB cameras.

## 🛠️ Tech Stack

*   **Language:** Python 3.x
*   **Web Framework:** Flask
*   **Computer Vision:** OpenCV (`cv2`)
*   **AI Engine:** TensorFlow Lite Runtime
*   **Model:** EfficientDet-Lite4 (COCO Dataset)

## 📦 Installation

### General (Linux/macOS/PC)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/edge-vision-ai.git
    cd edge-vision-ai
    ```

2.  **Install Dependencies:**
    ```bash
    pip install opencv-python flask numpy
    # You will also need tflite-runtime. Check https://www.tensorflow.org/lite/guide/python for your specific platform.
    ```

3.  **Download Model:**
    Ensure `efficientdet_lite4.tflite` and `coco_labels.txt` are in the project root.

### Android (Termux) Setup 📱

This project is specifically optimized for Android devices using Termux to turn old phones into powerful AI cameras.

👉 **See the [Detailed Setup Guide](SETUP_GUIDE.md) for step-by-step mobile instructions.**

## 🚦 Usage

1.  **Start your video source:**
    *   **IP Webcam (Android):** Start the server on your phone app.
    *   **USB Cam:** Connect your camera.

2.  **Run the Server:**
    ```bash
    python detect_server.py --stream <VIDEO_SOURCE>
    ```

    *Example (IP Webcam):*
    ```bash
    python detect_server.py --stream http://192.168.1.100:8080/video
    ```

    *Example (Local Webcam):*
    ```bash
    python detect_server.py --stream 0
    ```

3.  **Access the Dashboard:**
    Open your browser and navigate to:
    `http://localhost:5000` (or `http://<DEVICE_IP>:5000` if accessing remotely)

## ⚙️ Configuration

You can customize the behavior with command-line arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--stream` | `http://127.0.0.1:8080/video` | Video source URL or ID |
| `--model` | `efficientdet_lite4.tflite` | Path to TFLite model file |
| `--labels` | `coco_labels.txt` | Path to labels file |
| `--threshold`| `0.5` | Detection confidence threshold (0.0 - 1.0) |
| `--port` | `5000` | Web server port |

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
