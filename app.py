import os
import cv2
import base64
import pickle
import numpy as np
import requests
import time
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# MQTT Client
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠️  paho-mqtt not installed. MQTT features disabled.")

try:
    import face_recognition
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False

app = Flask(__name__)

# Configure max content length (50MB) to handle large base64 images
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Enable CORS with explicit configuration
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"]
     }},
     supports_credentials=True
)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

KNOWN_FACES_DIR = "./known_faces"
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.45'))
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ESP32_CAM_IP = os.getenv('ESP32_CAM_IP', '10.124.88.102')
ESP32_CAM_RESOLUTION = '640x480'
ESP32_CAM_TIMEOUT = int(os.getenv('ESP32_CAM_TIMEOUT', '10'))

# Auto recognition settings
AUTO_RECOGNITION = os.getenv('AUTO_RECOGNITION', 'true').lower() == 'true'
AUTO_RECOGNITION_INTERVAL = int(os.getenv('AUTO_RECOGNITION_INTERVAL', '3'))  # seconds
AUTO_UNLOCK_COOLDOWN = int(os.getenv('AUTO_UNLOCK_COOLDOWN', '8'))  # seconds min gap between unlock publish

# Thread control
auto_recognition_thread = None
auto_recognition_stop_event = threading.Event()
last_auto_unlock_ts = 0.0

BACKEND_GO_URL = os.getenv('BACKEND_GO_URL', 'http://10.124.88.57:8080')
# Safeguard common typo
if 'loccalhost' in BACKEND_GO_URL:
    BACKEND_GO_URL = BACKEND_GO_URL.replace('loccalhost', 'localhost')

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', 'broker.hivemq.com')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_TOPIC_PREFIX = 'iotcihuy/home'

# Initialize MQTT Client
mqtt_client = None
if MQTT_AVAILABLE:
    mqtt_client = mqtt.Client(client_id="python_face_service")
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"✅ MQTT Connected: {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"⚠️  MQTT Connection failed: {e}")
        mqtt_client = None


known_face_encodings = []
known_face_names = []
known_face_ids = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_known_faces():
    global known_face_encodings, known_face_names, known_face_ids
    
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return 0
    
    count = 0
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.pkl'):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    known_face_encodings.append(data['encoding'])
                    known_face_names.append(data['name'])
                    known_face_ids.append(data['user_id'])
                    count += 1
            except Exception as e:
                pass
    
    return count


def decode_base64_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"❌ [decode_base64_image] Failed to decode image - cv2.imdecode returned None")
            return None
            
        print(f"✅ [decode_base64_image] Image decoded successfully: shape={img.shape}, dtype={img.dtype}")
        return img
    except Exception as e:
        print(f"❌ [decode_base64_image] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def recognize_face_dlib(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        print(f"❌ [RECOGNIZE] No face detected in image")
        return None, None, 0.0, "No face detected"
    
    print(f"✅ [RECOGNIZE] Detected {len(face_locations)} face(s)")
    
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        print(f"❌ [RECOGNIZE] Could not encode face")
        return None, None, 0.0, "Could not encode face"
    
    for face_encoding in face_encodings:
        if not known_face_encodings:
            print(f"⚠️  [RECOGNIZE] No known faces enrolled yet")
            return None, None, 0.0, "No known faces enrolled"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]
        confidence = 1 - best_distance
        
        print(f" [RECOGNIZE] Best match distance: {best_distance:.4f}, Confidence: {confidence:.4f} (Threshold: {CONFIDENCE_THRESHOLD})")
        
        if confidence >= CONFIDENCE_THRESHOLD:
            recognized_name = known_face_names[best_match_idx]
            print(f" [RECOGNIZE] MATCH! {recognized_name} (confidence: {confidence:.4f})")
            return (
                known_face_ids[best_match_idx],
                recognized_name,
                confidence,
                "Face recognized"
            )
    
    print(f" [RECOGNIZE] No face matched confidence threshold")
    return None, None, 0.0, "Face not recognized"


def enroll_face_dlib(image, user_id, name, sample_number=1):
    print(f"[enroll_face_dlib] Processing image for user {user_id}")
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("[enroll_face_dlib] Detecting face locations...")
        face_locations = face_recognition.face_locations(rgb_image)
        print(f"[enroll_face_dlib] Found {len(face_locations)} face(s)")
        
        if not face_locations:
            print("[enroll_face_dlib] No face detected")
            return False, "No face detected in image"
        
        if len(face_locations) > 1:
            print("[enroll_face_dlib] Multiple faces detected")
            return False, "Multiple faces detected"
        
        print("[enroll_face_dlib] Encoding face...")
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    except Exception as e:
        print(f"[enroll_face_dlib] Exception during face detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Error during face detection: {str(e)}"
    
    if not face_encodings:
        print("[enroll_face_dlib] Could not encode face")
        return False, "Could not encode face"
    
    face_encoding = face_encodings[0]
    print("[enroll_face_dlib] Face encoded successfully")
    
    data = {
        'user_id': user_id,
        'name': name,
        'encoding': face_encoding,
        'sample_number': sample_number,
        'enrolled_at': datetime.now().isoformat()
    }
    
    filename = f"{user_id}_{name.replace(' ', '_')}_{sample_number}.pkl"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    print(f"[enroll_face_dlib] Saving to: {filepath}")
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[enroll_face_dlib] File saved successfully")
    except Exception as e:
        print(f"[enroll_face_dlib] Error saving file: {str(e)}")
        return False, f"Error saving face data: {str(e)}"
    
    print("[enroll_face_dlib] Reloading known faces...")
    load_known_faces()
    print(f"[enroll_face_dlib] Success! Total known faces: {len(known_face_names)}")
    
    return True, f"Face enrolled successfully"


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'service': 'face-recognition',
        'status': 'running',
        'face_recognition_lib': USE_FACE_RECOGNITION,
        'known_faces_count': len(known_face_names)
    })


@app.route('/validate-face', methods=['POST'])
def validate_face():
    print("\n" + "="*50)
    print("[VALIDATE-FACE] Request received")
    try:
        data = request.get_json()
        print(f"[VALIDATE-FACE] Data received: {data is not None}")
        
        if not data or 'image' not in data:
            print("[VALIDATE-FACE] ERROR: No image in request")
            return jsonify({'valid': False, 'faces_detected': 0, 'message': 'No image provided'}), 400
        
        print(f"[VALIDATE-FACE] Image data length: {len(data['image'])} characters")
        image = decode_base64_image(data['image'])
        
        if image is None:
            print("[VALIDATE-FACE] ERROR: Failed to decode image")
            return jsonify({'valid': False, 'faces_detected': 0, 'message': 'Invalid image format'}), 400
        
        if not USE_FACE_RECOGNITION:
            print("[VALIDATE-FACE] ERROR: face_recognition library not available")
            return jsonify({'valid': False, 'faces_detected': 0, 'message': 'Face recognition library not available'}), 500
        
        print(f"[VALIDATE-FACE] Image shape: {image.shape}")
        print("[VALIDATE-FACE] Converting to RGB...")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("[VALIDATE-FACE] Detecting faces...")
        face_locations = face_recognition.face_locations(rgb_image)
        print(f"[VALIDATE-FACE] Faces detected: {len(face_locations)}")
        
        faces_count = len(face_locations)
        
        if faces_count == 0:
            print("[VALIDATE-FACE] RESULT: No face detected")
            return jsonify({
                'valid': False,
                'faces_detected': 0,
                'message': 'No face detected. Ensure good lighting and face is clearly visible'
            }), 200
        
        if faces_count > 1:
            print(f"[VALIDATE-FACE] RESULT: Multiple faces ({faces_count})")
            return jsonify({
                'valid': False,
                'faces_detected': faces_count,
                'message': f'Multiple faces detected ({faces_count}). Only 1 face is allowed'
            }), 200
        
        print("[VALIDATE-FACE] SUCCESS: 1 valid face detected")
        print("="*50 + "\n")
        return jsonify({
            'valid': True,
            'faces_detected': 1,
            'message': 'Face validation passed'
        }), 200
        
    except Exception as e:
        print(f"[VALIDATE-FACE] EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'valid': False, 'faces_detected': 0, 'message': str(e)}), 500


@app.route('/enroll-base64', methods=['POST'])
def enroll_base64():
    print("\n" + "="*50)
    print("[ENROLL-BASE64] Request received")
    try:
        data = request.get_json()
        print(f"[ENROLL-BASE64] Data received: {data is not None}")
        
        if not data:
            print("[ENROLL-BASE64] ERROR: No data provided")
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['user_id', 'name', 'image']
        for field in required_fields:
            if field not in data:
                print(f"[ENROLL-BASE64] ERROR: Missing field: {field}")
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        user_id = data['user_id']
        name = data['name']
        print(f"[ENROLL-BASE64] User ID: {user_id}, Name: {name}")
        print(f"[ENROLL-BASE64] Image data length: {len(data['image'])} characters")
        
        image = decode_base64_image(data['image'])
        
        if image is None:
            print("[ENROLL-BASE64] ERROR: Failed to decode image")
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400
        
        if not USE_FACE_RECOGNITION:
            print("[ENROLL-BASE64] ERROR: face_recognition library not available")
            return jsonify({'success': False, 'error': 'Face recognition library not available'}), 500
        
        print(f"[ENROLL-BASE64] Image shape: {image.shape}")
        existing_count = len([f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(f"{user_id}_")])
        sample_number = existing_count + 1
        print(f"[ENROLL-BASE64] Sample number: {sample_number}")
        
        print("[ENROLL-BASE64] Starting enrollment process...")
        success, message = enroll_face_dlib(image, user_id, name, sample_number)
        
        filename = f"{user_id}_{name.replace(' ', '_')}_{sample_number}.pkl"
        
        if success:
            print(f"[ENROLL-BASE64] SUCCESS: {filename}")
            print("="*50 + "\n")
            return jsonify({
                'success': True,
                'message': message,
                'file': filename  # Changed from 'filename' to 'file' to match Go client
            }), 200
        else:
            print(f"[ENROLL-BASE64] FAILED: {message}")
            print("="*50 + "\n")
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        print(f"[ENROLL-BASE64] EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize face from Base64 image (called by backend Go)
    This endpoint is called from /api/face/recognize (backend Go)
    Backend Go handles access logging, so we only do recognition + MQTT here
    Request: { "image": "data:image/jpeg;base64,..." }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            print("❌ [API /recognize] No image provided")
            return jsonify({'success': False, 'recognized': False, 'error': 'No image provided'}), 400
        
        print("📥 [API /recognize] Received request from backend Go")
        image = decode_base64_image(data['image'])
        
        if image is None:
            print("❌ [API /recognize] Invalid/corrupt image format")
            return jsonify({'success': False, 'recognized': False, 'error': 'Invalid image format'}), 400
        
        print(f"✅ [API /recognize] Image decoded successfully: {image.shape}")
        
        if USE_FACE_RECOGNITION:
            user_id, name, confidence, message = recognize_face_dlib(image)
        else:
            user_id, name, confidence, message = None, None, 0, "Face recognition not available"
        
        recognized = user_id is not None
        
        print(f"📊 [API /recognize] Recognition result: recognized={recognized}, user_id={user_id}, name={name}")
        
        # 🔌 PUBLISH MQTT - Control door/buzzer based on recognition result
        if mqtt_client:
            if recognized:
                # UNLOCK DOOR
                door_payload = json.dumps({"action": "unlock", "user_id": user_id, "name": name})
                mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/door/control", door_payload)
                print(f"✅ [MQTT] Published door unlock: {door_payload} to {MQTT_TOPIC_PREFIX}/door/control")
                print(f"✅ [Note] Backend Go will handle access log recording for user {name}")
            else:
                # BUZZER WARNING
                buzzer_payload = json.dumps({"action": "on", "duration": 1000})
                mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/buzzer/control", buzzer_payload)
                print(f"⚠️  [MQTT] Published buzzer warning: {buzzer_payload} to {MQTT_TOPIC_PREFIX}/buzzer/control")
                print(f"✅ [Note] Backend Go will handle failed access log recording")
        else:
            print(f"❌ [MQTT] MQTT client not connected! Door control will NOT work!")
        
        response = {
            'success': True,
            'recognized': recognized,
            'user_id': user_id,
            'name': name,
            'confidence': round(confidence, 4) if confidence else 0,
            'message': message,
            'mqtt_connected': mqtt_client is not None
        }
        
        print(f"📤 [API /recognize] Response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ [API /recognize] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'recognized': False, 'error': str(e)}), 500


@app.route('/reload', methods=['POST'])
def reload_faces():
    try:
        count = load_known_faces()
        return jsonify({'success': True, 'message': f'Reloaded {count} faces'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/faces', methods=['GET'])
def list_faces():
    faces = []
    for i, name in enumerate(known_face_names):
        faces.append({
            'user_id': known_face_ids[i],
            'name': name
        })
    
    return jsonify({'success': True, 'count': len(faces), 'faces': faces})


@app.route('/faces/<int:user_id>', methods=['DELETE'])
def delete_face(user_id):
    try:
        deleted = False
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.startswith(f"{user_id}_") and filename.endswith('.pkl'):
                filepath = os.path.join(KNOWN_FACES_DIR, filename)
                os.remove(filepath)
                deleted = True
        
        if deleted:
            load_known_faces()
            return jsonify({'success': True, 'message': f'Face deleted for user_id: {user_id}'})
        else:
            return jsonify({'success': False, 'error': f'No face found for user_id: {user_id}'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def fetch_image_from_esp32cam(cam_ip=None, resolution=None):
    """Fetch single JPEG frame from ESP32-CAM with retry/fallback path."""
    if cam_ip is None:
        cam_ip = ESP32_CAM_IP
    if resolution is None:
        resolution = ESP32_CAM_RESOLUTION

    primary_url = f"http://{cam_ip}/{resolution}.jpg"
    fallback_url = f"http://{cam_ip}/capture"

    for attempt, url in enumerate([primary_url, fallback_url], start=1):
        try:
            response = requests.get(url, timeout=ESP32_CAM_TIMEOUT)

            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is not None:
                    if attempt > 1:
                        print(f"ℹ️  Fallback URL succeeded: {url}")
                    return image, None
                else:
                    err = "Failed to decode image"
                    print(f"⚠️  Decode failed (attempt {attempt}, url={url}): {err}")
            else:
                err = f"HTTP error: {response.status_code}"
                print(f"⚠️  HTTP error (attempt {attempt}, url={url}): {err}")

        except requests.exceptions.Timeout:
            err = "Connection timeout"
            print(f"⚠️  Timeout (attempt {attempt}, url={url})")
        except requests.exceptions.ConnectionError:
            err = f"Cannot connect to {cam_ip}"
            print(f"⚠️  Connection error (attempt {attempt}, url={url})")
        except Exception as e:
            err = str(e)
            print(f"⚠️  Exception (attempt {attempt}, url={url}): {err}")

        # If first attempt failed, try fallback; otherwise return last error
    return None, err


def auto_recognition_loop():
    """Background loop pulling frames from ESP32-CAM for auto unlock."""
    print(f"🚀 Auto recognition loop started (every {AUTO_RECOGNITION_INTERVAL}s)")
    while not auto_recognition_stop_event.is_set():
        start_ts = time.time()

        if not USE_FACE_RECOGNITION:
            print("⚠️  Auto recognition skipped: face_recognition not available")
            auto_recognition_stop_event.wait(AUTO_RECOGNITION_INTERVAL)
            continue

        frame, err = fetch_image_from_esp32cam()
        if frame is None:
            print(f"⚠️  Auto recognition: failed to fetch frame ({err})")
            auto_recognition_stop_event.wait(AUTO_RECOGNITION_INTERVAL)
            continue

        user_id, name, confidence, message = recognize_face_dlib(frame)
        recognized = user_id is not None

        print(f"🔍 Auto recognition result: recognized={recognized}, name={name}, conf={confidence:.4f}, msg={message}")

        if mqtt_client:
            if recognized:
                global last_auto_unlock_ts
                now_ts = time.time()
                if now_ts - last_auto_unlock_ts < AUTO_UNLOCK_COOLDOWN:
                    remaining = AUTO_UNLOCK_COOLDOWN - (now_ts - last_auto_unlock_ts)
                    print(f"⏳ Auto unlock cooldown active ({remaining:.1f}s left); skipping unlock publish")
                else:
                    # Send both command/action for compatibility with ESP32 & Go backend
                    door_payload_dict = {
                        "command": "unlock",
                        "action": "unlock",
                        "user_id": user_id,
                        "name": name,
                        "source": "auto"
                    }
                    door_payload = json.dumps(door_payload_dict)
                    mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/door/control", door_payload)
                    last_auto_unlock_ts = now_ts
                    print(f"✅ [MQTT] Auto door unlock published: {door_payload}")

                    # Log access to backend
                    try:
                        requests.post(f"{BACKEND_GO_URL}/api/access-log", json={
                            "user_id": user_id,
                            "method": "face",
                            "status": "success",
                            "message": "auto-recognition"
                        }, timeout=3)
                        print("✅ [Backend Log] Auto recognition success logged")
                    except Exception as log_err:
                        print(f"⚠️  [Backend Log] Failed to log auto recognition: {log_err}")
            else:
                # Unknown face; avoid buzzer spam in loop
                print("⚠️  Auto recognition: face not recognized (no buzzer)")
        else:
            print("❌ [MQTT] MQTT client not connected; auto unlock not sent")

        elapsed = time.time() - start_ts
        wait_time = max(0, AUTO_RECOGNITION_INTERVAL - elapsed)
        auto_recognition_stop_event.wait(wait_time)


def start_auto_recognition_thread():
    global auto_recognition_thread
    if not AUTO_RECOGNITION:
        print("ℹ️  Auto recognition disabled via AUTO_RECOGNITION=false")
        return
    if auto_recognition_thread and auto_recognition_thread.is_alive():
        print("ℹ️  Auto recognition thread already running")
        return
    auto_recognition_stop_event.clear()
    auto_recognition_thread = threading.Thread(target=auto_recognition_loop, daemon=True)
    auto_recognition_thread.start()


def stop_auto_recognition_thread():
    auto_recognition_stop_event.set()
    if auto_recognition_thread:
        auto_recognition_thread.join(timeout=1)
        print("🛑 Auto recognition thread stopped")


@app.route('/cam-proxy', methods=['GET'])
def cam_proxy():
    cam_ip = request.args.get('ip', ESP32_CAM_IP)
    resolution = request.args.get('resolution', ESP32_CAM_RESOLUTION)
    
    try:
        url = f"http://{cam_ip}/{resolution}.jpg"
        response = requests.get(url, timeout=ESP32_CAM_TIMEOUT)
        
        if response.status_code == 200:
            return Response(
                response.content,
                mimetype='image/jpeg',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        else:
            return jsonify({'error': 'Failed to fetch image'}), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Camera timeout'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': f'Cannot connect to camera at {cam_ip}'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recognize-cam', methods=['GET', 'POST'])
def recognize_from_cam():
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
            cam_ip = data.get('cam_ip', ESP32_CAM_IP)
            resolution = data.get('resolution', ESP32_CAM_RESOLUTION)
        else:
            cam_ip = request.args.get('ip', ESP32_CAM_IP)
            resolution = request.args.get('resolution', ESP32_CAM_RESOLUTION)
        
        image, error = fetch_image_from_esp32cam(cam_ip, resolution)
        
        if image is None:
            return jsonify({
                'success': False,
                'recognized': False,
                'error': error or 'Failed to fetch image',
                'cam_ip': cam_ip
            }), 500
        
        if USE_FACE_RECOGNITION:
            user_id, name, confidence, message = recognize_face_dlib(image)
        else:
            user_id, name, confidence, message = None, None, 0, "Face recognition not available"
        
        recognized = user_id is not None
        
        return jsonify({
            'success': True,
            'recognized': recognized,
            'user_id': user_id,
            'name': name,
            'confidence': round(confidence, 4) if confidence else 0,
            'message': message,
            'cam_ip': cam_ip
        })
        
    except Exception as e:
        return jsonify({'success': False, 'recognized': False, 'error': str(e)}), 500


@app.route('/recognize-base64', methods=['POST', 'OPTIONS'])
def recognize_base64():
    """
    Recognize face from Base64 image (for testing & web frontend)
    Request: { "image": "data:image/jpeg;base64,..." }
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        print("🔄 [API /recognize-base64] CORS preflight request")
        return jsonify({'success': True}), 200
    
    print("\n" + "="*80)
    print("📥 [API /recognize-base64] ===== NEW REQUEST =====")
    print(f"   Method: {request.method}")
    print(f"   Content-Type: {request.content_type}")
    print(f"   Remote Address: {request.remote_addr}")
    
    try:
        data = request.get_json()
        
        if not data:
            print("❌ [API /recognize-base64] No JSON data received")
            return jsonify({
                'success': False,
                'recognized': False,
                'message': 'No JSON data'
            }), 400
            
        if 'image' not in data:
            print("❌ [API /recognize-base64] 'image' field not found in request")
            return jsonify({
                'success': False,
                'recognized': False,
                'message': 'No image provided'
            }), 400
        
        image_data = data['image']
        print(f"✅ [API /recognize-base64] Image data received, length: {len(image_data)}")
        print("📥 [API /recognize-base64] Decoding base64...")
        image = decode_base64_image(data['image'])
        
        if image is None:
            print("❌ [API /recognize-base64] Invalid/corrupt image format")
            return jsonify({
                'success': False,
                'recognized': False,
                'message': 'Invalid image format'
            }), 400
        
        print(f"✅ [API /recognize-base64] Image decoded successfully: {image.shape}")
        
        if not USE_FACE_RECOGNITION:
            print("❌ [API /recognize-base64] Face recognition library not available")
            return jsonify({
                'success': False,
                'recognized': False,
                'message': 'Face recognition library not available'
            }), 500
        
        user_id, name, confidence, message = recognize_face_dlib(image)
        
        recognized = user_id is not None
        
        print(f"📊 [API /recognize-base64] Recognition result: recognized={recognized}, user_id={user_id}, name={name}")
        
        # 🔌 PUBLISH MQTT - Control door/buzzer based on recognition result
        if mqtt_client:
            if recognized:
                # UNLOCK DOOR
                door_payload = json.dumps({"action": "unlock", "user_id": user_id, "name": name})
                mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/door/control", door_payload)
                print(f"✅ [MQTT] Published door unlock: {door_payload} to {MQTT_TOPIC_PREFIX}/door/control")
                
                # Log access ke backend
                try:
                    requests.post(f"{BACKEND_GO_URL}/api/access-log", json={
                        "user_id": user_id,
                        "method": "face",
                        "status": "success"
                    }, timeout=3)
                    print(f"✅ [Backend Log] Access log recorded")
                except Exception as log_err:
                    print(f"⚠️  [Backend Log] Failed to log: {log_err}")
            else:
                # BUZZER WARNING
                buzzer_payload = json.dumps({"action": "on", "duration": 1000})
                mqtt_client.publish(f"{MQTT_TOPIC_PREFIX}/buzzer/control", buzzer_payload)
                print(f"⚠️  [MQTT] Published buzzer warning: {buzzer_payload} to {MQTT_TOPIC_PREFIX}/buzzer/control")
                
                # Log failed attempt
                try:
                    requests.post(f"{BACKEND_GO_URL}/api/access-log", json={
                        "method": "face",
                        "status": "failed",
                        "message": "Face not recognized"
                    }, timeout=3)
                    print(f"✅ [Backend Log] Failed access log recorded")
                except Exception as log_err:
                    print(f"⚠️  [Backend Log] Failed to log: {log_err}")
        else:
            print(f"❌ [MQTT] MQTT client not connected! Door control will NOT work!")
        
        response = {
            'success': True,
            'recognized': recognized,
            'user_id': user_id,
            'name': name,
            'confidence': round(confidence, 4) if confidence else 0,
            'message': message,
            'mqtt_connected': mqtt_client is not None
        }
        
        print(f"📤 [API /recognize-base64] Response: {response}")
        print("="*80 + "\n")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ [API /recognize-base64] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return jsonify({
            'success': False,
            'recognized': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    load_known_faces()
    start_auto_recognition_thread()
    app.run(host='0.0.0.0', port=5001, debug=True)
