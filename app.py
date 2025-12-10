import os
import cv2
import base64
import pickle
import numpy as np
import requests
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

try:
    import face_recognition
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False

app = Flask(__name__)
CORS(app)

KNOWN_FACES_DIR = "./known_faces"
CONFIDENCE_THRESHOLD = 0.65
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ESP32_CAM_IP = os.getenv('ESP32_CAM_IP', '10.124.88.102')
ESP32_CAM_RESOLUTION = '640x480'
ESP32_CAM_TIMEOUT = 10

BACKEND_GO_URL = os.getenv('BACKEND_GO_URL', 'http://10.124.88.57:8080')

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
        return img
    except Exception as e:
        return None


def recognize_face_dlib(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return None, None, 0.0, "No face detected"
    
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        return None, None, 0.0, "Could not encode face"
    
    for face_encoding in face_encodings:
        if not known_face_encodings:
            return None, None, 0.0, "No known faces enrolled"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]
        confidence = 1 - best_distance
        
        if confidence >= CONFIDENCE_THRESHOLD:
            return (
                known_face_ids[best_match_idx],
                known_face_names[best_match_idx],
                confidence,
                "Face recognized"
            )
    
    return None, None, 0.0, "Face not recognized"


def enroll_face_dlib(image, user_id, name, sample_number=1):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return False, "No face detected in image"
    
    if len(face_locations) > 1:
        return False, "Multiple faces detected"
    
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        return False, "Could not encode face"
    
    face_encoding = face_encodings[0]
    
    data = {
        'user_id': user_id,
        'name': name,
        'encoding': face_encoding,
        'sample_number': sample_number,
        'enrolled_at': datetime.now().isoformat()
    }
    
    filename = f"{user_id}_{name.replace(' ', '_')}_{sample_number}.pkl"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    load_known_faces()
    
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
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'valid': False, 'faces_detected': 0, 'message': 'No image provided'}), 400
        
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({'valid': False, 'faces_detected': 0, 'message': 'Invalid image format'}), 400
        
        if not USE_FACE_RECOGNITION:
            return jsonify({'valid': False, 'faces_detected': 0, 'message': 'Face recognition library not available'}), 500
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        faces_count = len(face_locations)
        
        if faces_count == 0:
            return jsonify({
                'valid': False,
                'faces_detected': 0,
                'message': 'No face detected. Ensure good lighting and face is clearly visible'
            }), 200
        
        if faces_count > 1:
            return jsonify({
                'valid': False,
                'faces_detected': faces_count,
                'message': f'Multiple faces detected ({faces_count}). Only 1 face is allowed'
            }), 200
        
        return jsonify({
            'valid': True,
            'faces_detected': 1,
            'message': 'Face validation passed'
        }), 200
        
    except Exception as e:
        return jsonify({'valid': False, 'faces_detected': 0, 'message': str(e)}), 500


@app.route('/enroll-base64', methods=['POST'])
def enroll_base64():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required_fields = ['user_id', 'name', 'image']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        user_id = data['user_id']
        name = data['name']
        
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400
        
        if not USE_FACE_RECOGNITION:
            return jsonify({'success': False, 'error': 'Face recognition library not available'}), 500
        
        existing_count = len([f for f in os.listdir(KNOWN_FACES_DIR) if f.startswith(f"{user_id}_")])
        sample_number = existing_count + 1
        
        success, message = enroll_face_dlib(image, user_id, name, sample_number)
        
        filename = f"{user_id}_{name.replace(' ', '_')}_{sample_number}.pkl"
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'file': filename  # Changed from 'filename' to 'file' to match Go client
            }), 200
        else:
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'recognized': False, 'error': 'No image provided'}), 400
        
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({'success': False, 'recognized': False, 'error': 'Invalid image format'}), 400
        
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
            'message': message
        })
        
    except Exception as e:
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
    if cam_ip is None:
        cam_ip = ESP32_CAM_IP
    if resolution is None:
        resolution = ESP32_CAM_RESOLUTION
    
    url = f"http://{cam_ip}/{resolution}.jpg"
    
    try:
        response = requests.get(url, timeout=ESP32_CAM_TIMEOUT)
        
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                return image, None
            else:
                return None, "Failed to decode image"
        else:
            return None, f"HTTP error: {response.status_code}"
                
    except requests.exceptions.Timeout:
        return None, "Connection timeout"
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to {cam_ip}"
    except Exception as e:
        return None, str(e)


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


if __name__ == '__main__':
    load_known_faces()
    app.run(host='0.0.0.0', port=5001, debug=True)
