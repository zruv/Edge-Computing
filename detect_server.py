import cv2
import numpy as np
import time
import argparse
import threading
from flask import Flask, Response, render_template_string
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tflite_runtime.interpreter import Interpreter
    load_delegate = None

# Initialize Flask
app = Flask(__name__)

# --- Global Shared Variables ---
# These are shared between the Camera Thread, AI Thread, and Web Server
current_frame = None          # The latest frame from the camera (Raw)
current_detections = []       # The latest results from the AI
frame_lock = threading.Lock() # Thread safety for frame access
detections_lock = threading.Lock() # Thread safety for detections access
program_running = True        # To stop threads cleanly
ai_status = {"model": "Loading...", "mode": "Initializing...", "status": "WAITING"}

# Configuration
class DefaultArgs:
    device_name = "EDGE-DEVICE"
    stream = "0"
    model = "efficientdet_lite4.tflite"
    labels = "coco_labels.txt"
    threshold = 0.3
    enable_npu = False
    port = 5000

args = DefaultArgs()

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
    
    # Process stream argument
    stream_source = args.stream
    if isinstance(stream_source, str):
        # If it's a number (e.g., "0"), convert to int for USB camera
        if stream_source.isdigit():
            stream_source = int(stream_source)
        # If it looks like an IP but misses http://, add it
        elif "http" not in stream_source and (":" in stream_source or "." in stream_source):
             stream_source = "http://" + stream_source
             print(f"Auto-corrected stream URL to: {stream_source}")

    print(f"Connecting to camera stream at {stream_source}...")
    cap = cv2.VideoCapture(stream_source)
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
    
    # Configure NPU Delegate if requested
    delegates = []
    if args.enable_npu:
        print("Attempting to load NNAPI (NPU) Delegate...")
        if load_delegate is not None:
            # List of potential paths for Android NNAPI
            possible_lib_paths = [
                'libnnapi.so',               # Standard lookup
                '/system/lib64/libnnapi.so', # Android 64-bit system (Common)
                '/system/lib/libnnapi.so'    # Android 32-bit system
            ]
            
            delegate = None
            for lib_path in possible_lib_paths:
                try:
                    delegate = load_delegate(lib_path)
                    print(f"SUCCESS: NNAPI Delegate loaded from {lib_path}!")
                    break
                except Exception:
                    continue
            
            if delegate:
                delegates = [delegate]
            else:
                print("WARNING: Failed to load NNAPI delegate from any known path (libnnapi.so).")
                print("Falling back to CPU.")
        else:
            print("WARNING: 'load_delegate' not found in tflite_runtime. Cannot use NPU.")

    try:
        interpreter = Interpreter(model_path=args.model, experimental_delegates=delegates)
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
    
    # Status for Dashboard
    global ai_status
    accel_type = "NNAPI (HW Accel)" if args.enable_npu else "CPU (Standard)"
    
    ai_status = {
        "model": args.model.split('/')[-1],
        "mode": accel_type,
        "status": "ACTIVE"
    }

    print(f"AI Engine Started on {args.device_name} using {accel_type}")

    while program_running:
        # 1. Get the latest frame
        frame_to_process = None
        with frame_lock:
            if current_frame is not None:
                frame_to_process = current_frame.copy()
        
        if frame_to_process is None:
            time.sleep(0.1)
            continue

        # 2. Pre-process (Letterbox Resize to fix stretching)
        h_orig, w_orig = frame_to_process.shape[:2]
        
        # Calculate scale to fit inside model input while keeping aspect ratio
        scale = min(width / w_orig, height / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        
        # Resize the image
        resized_image = cv2.resize(frame_to_process, (new_w, new_h))
        
        # Create a blank canvas (gray background) of model size
        input_data = np.full((height, width, 3), 128, dtype=np.uint8)
        
        # Paste the resized image into the center
        dw = (width - new_w) // 2
        dh = (height - new_h) // 2
        input_data[dh:dh+new_h, dw:dw+new_w] = resized_image
        
        # Expand dims for model input
        input_data = np.expand_dims(input_data, axis=0)

        if is_floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # EfficientDet usually expects uint8, but just in case
            input_data = input_data.astype(input_details[0]['dtype'])

        # 3. Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # 4. Process Results
        new_detections = []
        for i in range(len(scores)):
            if scores[i] > args.threshold:
                class_id = int(classes[i])
                object_name = labels[class_id] if class_id < len(labels) else "Unknown"

                # FILTER: Only show Humans
                if object_name != 'person':
                    continue

                # Get box in model coordinates (0.0 - 1.0)
                ymin, xmin, ymax, xmax = boxes[i]

                # Convert Model Coords -> Original Image Coords
                # 1. Denormalize to model pixel coords
                xmin = xmin * width
                xmax = xmax * width
                ymin = ymin * height
                ymax = ymax * height

                # 2. Remove padding (shift origin)
                xmin -= dw
                xmax -= dw
                ymin -= dh
                ymax -= dh

                # 3. Scale back to original size
                xmin /= scale
                xmax /= scale
                ymin /= scale
                ymax /= scale

                # 4. Normalize back to 0.0 - 1.0 for the display thread
                xmin = max(0, min(1, xmin / w_orig))
                xmax = max(0, min(1, xmax / w_orig))
                ymin = max(0, min(1, ymin / h_orig))
                ymax = max(0, min(1, ymax / h_orig))

                # Filter out bad boxes
                if ymin >= ymax or xmin >= xmax: continue
                
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
    return render_template_string("""
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
                    <div class="info-value">{{ model }}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Hardware</div>
                    <div class="info-value">{{ mode }}</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Status</div>
                    <div class="info-value">{{ status }}</div>
                </div>
            </div>
        </div>

        <footer>
            SECURE CONNECTION • {{ device }} • TERMUX
        </footer>
    </body>
    </html>
    """, model=ai_status['model'], mode=ai_status['mode'], status=ai_status['status'], device=args.device_name)

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
    parser.add_argument('--enable_npu', action='store_true', help='Attempt to use NNAPI (NPU/GPU) acceleration')
    
    # Auto-detect device name if possible (works on Android Termux)
    default_device = 'ANDROID-EDGE'
    try:
        import subprocess
        prop = subprocess.check_output(['getprop', 'ro.product.model']).decode('utf-8').strip()
        if prop: default_device = prop
    except:
        pass

    parser.add_argument('--device_name', default=default_device, help='Device name for dashboard')
    
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