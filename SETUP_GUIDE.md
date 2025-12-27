# Edge Vision AI - Setup Guide (Samsung A30 Edition)

**Project:** High-Accuracy Object Detection on Android (via Termux)
**Model:** EfficientDet-Lite4 (State-of-the-Art Accuracy)
**Interface:** Modern Dark Mode Dashboard with Async Video
**Status:** **Optimized for Samsung A30**

This guide will help you set up your Android phone to run a powerful AI object detector that streams results to any device on your Wi-Fi network.

---

## 1. Prerequisites (On Your Phone)

You need to install two apps from the Google Play Store (or F-Droid):

1.  **Termux** (Terminal emulator)
    *   *Recommendation:* Download from F-Droid for the latest version.
2.  **IP Webcam** (For the camera stream)
    *   *Settings:* Open IP Webcam -> Video Preferences -> Set Resolution to **1280x720** or **640x480** (Do not use 1080p to save CPU).

---

## 2. Transfer Files

You need to move the `edge-ai` folder from your PC to your phone. The folder is now clean and contains only the 5 essential files:

*   **`detect_server.py`**: The main AI engine with the new Dashboard.
*   **`efficientdet_lite4.tflite`**: The high-accuracy AI model.
*   **`coco_labels.txt`**: The dictionary of 80 object types (Person, Car, etc.).
*   **`install_flask.sh`**: One-click installer for the web server.
*   **`setup_termux.sh`**: One-click installer for Python & OpenCV.

**How to Transfer:**
*   **USB:** Plug in phone -> Copy folder to `Internal Storage/edge-ai`.
*   **Zip:** Email it to yourself or use Quick Share.

---

## 3. Installation (Inside Termux)

Open the **Termux** app on your phone and run these commands one by one:

### Step A: Grant Storage Permission
```bash
termux-setup-storage
# Click "Allow" on the popup
```

### Step B: Go to the Folder
Navigate to where you copied the files.
```bash
cd /sdcard/edge-ai
# OR if you copied it to the root
cd edge-ai
```

### Step C: Install Dependencies
We have prepared scripts to make this easy.
```bash
# 1. Update Termux
pkg update -y && pkg upgrade -y

# 2. Run the Setup Script (Installs Python, OpenCV, etc.)
bash setup_termux.sh

# 3. Install Flask (The Web Server)
bash install_flask.sh
```

---

## 4. Running the AI

### Step A: Start Camera
1.  Open **IP Webcam** app.
2.  Scroll to bottom and tap **"Start Server"**.
3.  Note the IP address on the screen (e.g., `http://192.168.1.8:8080`).

### Step B: Start AI Server (In Termux)
Go back to Termux and run:

```bash
# Replace the URL with YOUR IP Webcam URL
python detect_server.py --stream http://192.168.1.8:8080/video
```

**What to Expect:**
*   **Initialization:** The `Lite4` model is heavy; it may take 15-20 seconds to load.
*   **Performance:** The video will stream at a smooth **30 FPS**.
*   **Detection:** The green boxes will update every 1-3 seconds. This is normal for a Samsung A30 running a heavy model.

---

## 5. The Dashboard

Open a web browser on your **Laptop, Tablet, or the Phone itself** and go to:

> **http://<PHONE_IP>:5000**

**New Features:**
*   **Hacker Theme:** A professional black & neon-green interface.
*   **Clean Video:** The "System Online" badge is now in the header, keeping your view unobstructed.
*   **Loading Overlay:** Shows initialization status so you know the AI is working.
*   **Responsive:** optimized for both mobile and desktop screens.

---

## 6. Troubleshooting

*   **"Killed" or App Crashes:**
    *   The `Lite4` model uses a lot of RAM. If Termux crashes, close other apps.
    *   If it persists, you might need to download a smaller model (like `EfficientDet-Lite1`).

*   **Video Lag:**
    *   Ensure IP Webcam is set to **720p or lower**. 1080p is too much data for the WiFi/CPU.

*   **"Address already in use":**
    *   The server didn't close properly. Restart Termux or run on a different port:
    *   `python detect_server.py --port 5005 ...`
