from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import requests
import cv2
import threading
import time
import numpy as np
import math
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# Initialize Flask
app = Flask(__name__)
CORS(app)

# --- PI 5 GPIO CONFIGURATION (BTS7960 & L293D) ---
pins = {}

def init_gpio():
    global pins
    try:
        if not pins:
            # BTS7960 requires two PWM pins per motor for Forward/Backward (RPWM/LPWM)
            
            # 1. TELESCOPIC ARM (BTS7960 #1)
            # Physical Pin 32 (PWM) and Pin 33 (PWM)
            pins['arm_rpwm'] = PWMOutputDevice(12) # Extension
            pins['arm_lpwm'] = PWMOutputDevice(13) # Retraction
            
            # 2. PLATFORM TRACK (BTS7960 #2)
            # Physical Pin 12 (PWM) and Pin 35 (PWM)
            pins['track_rpwm'] = PWMOutputDevice(18) # Clockwise
            pins['track_lpwm'] = PWMOutputDevice(19) # Counter-Clockwise
            
            # 3. CUTTER MOTOR (L293D)
            # Physical Pin 40 connects to L293D IN1 (IN2 grounded, EN1 to 5V)
            pins['cutter'] = DigitalOutputDevice(21) 
            
            print("✅ GPIO: Pi 5 BTS7960 & L293D Hardware Pins Online.")
    except Exception as e:
        print(f"⚠️ GPIO Warning: {e}")

init_gpio()

# --- VISION CONFIGURATION ---
ESP_IP = "192.168.4.1" 
WIDTH, HEIGHT = 1280, 720
camera_mode = "OFF"

class CameraStream:
    def __init__(self):
        self.stream = None
        self.frame = None
        self.stopped = True
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if not self.stopped: return self
            for i in [0, 2, 4, 1]: 
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.stream = cap
                    self.stopped = False
                    threading.Thread(target=self.update, daemon=True).start()
                    break
                cap.release()
        return self

    def update(self):
        while not self.stopped:
            if self.stream:
                ret, frame = self.stream.read()
                if ret:
                    with self.lock: self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock: return self.frame is not None, self.frame

    def stop(self):
        with self.lock:
            self.stopped = True
            if self.stream: self.stream.release()
            self.stream = None

global_stream = CameraStream()

def perform_ai_detection(frame):
    """Accurate Arecanut Detection: Color + Circularity Logic"""
    try:
        small = cv2.resize(frame, (640, 360))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        ranges = [
            ((np.array([5, 100, 80]), np.array([28, 255, 255])), "RIPE", (0, 165, 255)),
            ((np.array([35, 60, 40]), np.array([85, 255, 255])), "UNRIPE", (0, 255, 0))
        ]
        for (low, high), label, bgr in ranges:
            mask = cv2.inRange(hsv, low, high)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 800:
                    perimeter = cv2.arcLength(c, True)
                    if perimeter == 0: continue
                    # Circularity filter (closer to 1.0 is a perfect circle/nut)
                    circularity = 4 * math.pi * (area / (perimeter * perimeter))
                    if 0.35 < circularity < 1.1:
                        x, y, w, h = cv2.boundingRect(c)
                        x, y, w, h = x*2, y*2, w*2, h*2 
                        cv2.rectangle(frame, (x, y), (x+w, y+h), bgr, 3)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
    except: pass
    return frame

def gen_frames():
    while True:
        if camera_mode == "OFF":
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "OFFLINE", (WIDTH//2-100, HEIGHT//2), 1, 4, (80,80,80), 3)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
        else:
            success, frame = global_stream.read()
            if not success:
                frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, "SYNCING...", (WIDTH//2-120, HEIGHT//2), 1, 3, (80,80,80), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
            else:
                if camera_mode == "AI": frame = perform_ai_detection(frame)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

# --- ROUTES ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/mode/<mode>')
def set_mode(mode):
    global camera_mode
    camera_mode = mode
    if mode != "OFF": global_stream.start()
    else: global_stream.stop()
    return jsonify({"status": "ok"})

@app.route('/api/motors/<command>')
def control_pi_motors(command):
    """
    Motor Logic: BTS7960 (PWM/PWM) and L293D (Digital Output).
    """
    try:
        # 1. TELESCOPIC ARM (BTS7960 #1)
        if command == 'ARM_EXTEND':
            pins['arm_lpwm'].value = 0
            pins['arm_rpwm'].value = 0.85
        elif command == 'ARM_RETRACT':
            pins['arm_rpwm'].value = 0
            pins['arm_lpwm'].value = 0.85
        
        # 2. PLATFORM TRACK (BTS7960 #2)
        elif command == 'TRACK_CW':
            pins['track_lpwm'].value = 0
            pins['track_rpwm'].value = 0.6
        elif command == 'TRACK_CCW':
            pins['track_rpwm'].value = 0
            pins['track_lpwm'].value = 0.6
        
        # 3. CUTTER MOTOR (L293D)
        elif command == 'CUT_TOGGLE':
            pins['cutter'].toggle()
        
        # 4. IDLE / STOP ALL
        elif command == 'IDLE':
            pins['arm_rpwm'].value = 0
            pins['arm_lpwm'].value = 0
            pins['track_rpwm'].value = 0
            pins['track_lpwm'].value = 0
            
    except Exception as e:
        print(f"Motor Control Error: {e}")
    return jsonify({"status": "ok"})

@app.route('/api/esp/<command>')
def relay_to_esp(command):
    try:
        # Relay to ESP8266 (Climbing Base)
        requests.get(f"http://{ESP_IP}/{command.lower()}", timeout=0.2)
        return jsonify({"status": "ok"})
    except: return jsonify({"status": "offline"}), 503

if __name__ == '__main__':
    # Pi 5 is fast enough to handle multiple threads easily
    app.run(host='0.0.0.0', port=8080, threaded=True, use_reloader=False)
