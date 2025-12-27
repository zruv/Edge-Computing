import cv2
import numpy as np
import time
import argparse
import threading
from flask import Flask, Response
from tflite_runtime.interpreter import Interpreter

# Initialize Flask
app = Flask(__name__)

# --- Global Shared Variables ---
# These are shared between the Camera Thread, AI Thread, and Web Server
current_frame = None          # The latest frame from the camera (Raw)
current_detections = []       # The latest results from the AI
frame_lock = threading.Lock() # Thread safety for frame access
detections_lock = threading.Lock() # Thread safety for detections access
program_running = True        # To stop threads cleanly

# Configuration
args = None

def load_labels(path):
    try:
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

# --- Thread 1: Camera Capture ---
# Grabs frames as fast as possible (30+ FPS)
def camera_thread_func():
    global current_frame, program_running
    
    print(f"Connecting to camera stream at {args.stream}...")
    cap = cv2.VideoCapture(args.stream)
    time.sleep(1.0) # Warm up

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        program_running = False
        return

    while program_running:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed. Reconnecting in 2s...")
            time.sleep(2)
            cap = cv2.VideoCapture(args.stream)
            continue
        
        # Update the global frame immediately
        with frame_lock:
            current_frame = frame.copy()
        
        # Small sleep to yield CPU
        time.sleep(0.01)
    
    cap.release()

# --- Thread 2: AI Processor ---
# Runs as fast as it can, but doesn't block the video
def ai_thread_func():
    global current_frame, current_detections, program_running
    
    print(f"Loading model: {args.model}...")
    try:
        interpreter = Interpreter(model_path=args.model)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        program_running = False
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    is_floating_model = (input_details[0]['dtype'] == np.float32)

    labels = load_labels(args.labels)
    
    print("AI Engine Started (Async Mode)")

    while program_running:
        # 1. Get the latest frame
        frame_to_process = None
        with frame_lock:
            if current_frame is not None:
                frame_to_process = current_frame.copy()
        
        if frame_to_process is None:
            time.sleep(0.1)
            continue

        # 2. Pre-process
        frame_resized = cv2.resize(frame_to_process, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if is_floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = input_data.astype(input_details[0]['dtype'])

        # 3. Inference (This is the slow part)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # 4. Process Results
        new_detections = []
        for i in range(len(scores)):
            if scores[i] > args.threshold:
                # We save normalized coordinates (0.0 to 1.0) so we can scale them to any display size later
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Filter out bad boxes
                if ymin > ymax or xmin > xmax: continue

                class_id = int(classes[i])
                object_name = labels[class_id] if class_id < len(labels) else "Unknown"
                confidence = int(scores[i] * 100)
                
                new_detections.append({
                    'box': (ymin, xmin, ymax, xmax),
                    'label': object_name,
                    'score': confidence
                })

        # 5. Update global detections
        with detections_lock:
            current_detections = new_detections

# --- Thread 3: Video Stream Generator ---
# Combines latest frame + latest boxes for smooth display
def generate_frames():
    global current_frame, current_detections
    
    while program_running:
        # Get frame
        frame_display = None
        with frame_lock:
            if current_frame is None:
                time.sleep(0.05)
                continue
            frame_display = current_frame.copy()
        
        # Resize for streaming (16:9 aspect ratio, 640x360) for bandwidth efficiency
        # We process on full size, but stream smaller
        h_orig, w_orig = frame_display.shape[:2]
        frame_display = cv2.resize(frame_display, (640, 360))
        h_disp, w_disp = frame_display.shape[:2]

        # Get latest detections
        detections_to_draw = []
        with detections_lock:
            detections_to_draw = list(current_detections) # Copy list

        # Draw boxes
        for det in detections_to_draw:
            ymin, xmin, ymax, xmax = det['box']
            label = det['label']
            score = det['score']

            # Scale normalized coordinates to display size
            left = int(xmin * w_disp)
            top = int(ymin * h_disp)
            right = int(xmax * w_disp)
            bottom = int(ymax * h_disp)

            # Draw
            cv2.rectangle(frame_display, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame_display, f"{label} {score}%", (left, max(top - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode
        (flag, encodedImage) = cv2.imencode(".jpg", frame_display)
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        
        # Limit stream FPS slightly to save bandwidth (30 FPS max)
        time.sleep(0.033)

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Edge Vision AI</title>
        <style>
            :root {
                --bg-color: #000000; /* Pure Black */
                --surface-color: #0a0a0a; /* Almost Black */
                --border-color: #1f1f1f;
                --primary-color: #00ff41; /* Terminal Green */
                --secondary-color: #008f11; /* Darker Green */
                --text-main: #e0e0e0;
                --text-dim: #555;
            }
            body {
                margin: 0;
                padding: 0;
                background-color: var(--bg-color);
                color: var(--text-main);
                font-family: \"Courier New\", Courier, monospace; /* Tech font */
                display: flex;
                flex-direction: column;
                min-height: 100vh;
                align-items: center;
            }
            header {
                width: 100%;
                padding: 1rem 0;
                background-color: var(--surface-color);
                border-bottom: 1px solid var(--border-color);
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px;
                margin-bottom: 2rem;
            }
            header h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--primary-color);
                letter-spacing: 2px;
                text-transform: uppercase;
                text-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
            }
            /* Status Badge - Now in Header */
            .status-badge {
                background-color: rgba(0, 255, 65, 0.1);
                color: var(--primary-color);
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 8px;
                border: 1px solid var(--secondary-color);
            }
            .status-dot {
                width: 8px;
                height: 8px;
                background-color: var(--primary-color);
                border-radius: 50%;
                box-shadow: 0 0 8px var(--primary-color);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.4; }
                100% { opacity: 1; }
            }

            .container {
                width: 94%;
                max-width: 1100px;
                display: flex;
                flex-direction: column;
                align-items: center;
                flex-grow: 1;
            }
            .video-card {
                width: 100%;
                background-color: var(--surface-color);
                border: 1px solid var(--border-color);
                border-radius: 4px; /* Sharper corners */
                padding: 4px;
                box-shadow: 0 0 20px rgba(0, 255, 65, 0.05);
            }
            .video-wrapper {
                width: 100%;
                aspect-ratio: 16 / 9;
                background-color: #050505;
                border-radius: 2px;
                overflow: hidden;
                position: relative;
            }
            .video-wrapper img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
                opacity: 0;
                transition: opacity 0.5s ease;
            }
            .video-wrapper img.loaded {
                opacity: 1;
            }
            /* Loading State */
            .loading-overlay {
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: var(--primary-color);
                z-index: 1;
                font-family: monospace;
            }
            .spinner {
                width: 40px;
                height: 40px;
                border: 2px solid var(--surface-color);
                border-top-color: var(--primary-color);
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
                margin-bottom: 15px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            
            .info-panel {
                margin-top: 20px;
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                width: 100%;
            }
            .info-card {
                background: var(--surface-color);
                border: 1px solid var(--border-color);
                padding: 15px;
                text-align: center;
            }
            .info-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; }
            .info-value { font-size: 1rem; margin-top: 5px; color: var(--primary-color); font-weight: bold;}

            footer {
                margin-top: auto;
                padding: 2rem;
                font-size: 0.7rem;
                color: var(--text-dim);
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Mobile Adjustments */
            @media (max-width: 600px) {
                header { flex-direction: column; gap: 10px; }
                .info-panel { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>EDGE VISION</h1>
            <div class="status-badge">
                <div class="status-dot"></div> SYSTEM ONLINE
            </div>
        </header>
        
        <div class="container">
            <div class="video-card">
                <div class="video-wrapper">
                    <div class="loading-overlay" id="loader">
                        <div class="spinner"></div>
                        <span>INITIALIZING NEURAL NET...</span>
                    </div>
                    <img src="/video_feed" onload="this.classList.add('loaded'); document.getElementById('loader').style.display='none';" onerror="alert('Stream connection lost.')">
                </div>
            </div>

            <div class="info-panel">
                <div class="info-card">
                    <div class="info-label">Model</div>
                    <div class="info-value">EfficientDet-Lite4</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Mode</div>
                    <div class="info-value">ASYNC / HIGH-RES</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Status</div>
                    <div class="info-value">ACTIVE</div>
                </div>
            </div>
        </div>

        <footer>
            SECURE CONNECTION • LOCALHOST • TERMUX
        </footer>
    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Web Server')
    parser.add_argument('--stream', default='http://127.0.0.1:8080/video', help='IP Webcam URL')
    parser.add_argument('--model', default='efficientdet_lite4.tflite', help='Model path')
    parser.add_argument('--labels', default='coco_labels.txt', help='Labels path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    
    args = parser.parse_args()

    # Start Camera Thread
    t_cam = threading.Thread(target=camera_thread_func)
    t_cam.daemon = True
    t_cam.start()

    # Start AI Thread
    t_ai = threading.Thread(target=ai_thread_func)
    t_ai.daemon = True
    t_ai.start()

    # Start Web Server
    # Host='0.0.0.0' allows access from other devices on the network
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)