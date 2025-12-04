"""
Smart Home Face Recognition Service
====================================
API endpoints:
- POST /recognize  - Recognize face from image
- POST /enroll     - Enroll new face
- POST /reload     - Reload known faces
- GET  /health     - Health check
- GET  /faces      - List enrolled faces
"""

import os
import cv2
import base64
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Try to import face_recognition, fallback to haar cascade if not available
try:
    import face_recognition
    USE_FACE_RECOGNITION = True
    print("✅ Using face_recognition library (dlib)")
except ImportError:
    USE_FACE_RECOGNITION = False
    print("⚠️  face_recognition not available, using Haar Cascade (less accurate)")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
KNOWN_FACES_DIR = "./known_faces"
CONFIDENCE_THRESHOLD = 0.6  # Lower = more strict (0.0 - 1.0)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Global storage for known faces
known_face_encodings = []
known_face_names = []
known_face_ids = []

# Haar cascade for fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_known_faces():
    """Load all known faces from disk"""
    global known_face_encodings, known_face_names, known_face_ids
    
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print("📁 Known faces directory not found, creating...")
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
                    print(f"  ✅ Loaded: {data['name']} (ID: {data['user_id']})")
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
    
    print(f"📊 Total faces loaded: {count}")
    return count


def decode_base64_image(base64_string):
    """Decode base64 string to numpy array (image)"""
    try:
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        return None


def recognize_face_dlib(image):
    """Recognize face using face_recognition library (dlib)"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find faces in image
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return None, None, 0.0, "No face detected"
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        return None, None, 0.0, "Could not encode face"
    
    # Compare with known faces
    for face_encoding in face_encodings:
        if not known_face_encodings:
            return None, None, 0.0, "No known faces enrolled"
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Find best match
        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]
        
        # Convert distance to confidence (0-1, higher is better)
        confidence = 1 - best_distance
        
        if confidence >= CONFIDENCE_THRESHOLD:
            return (
                known_face_ids[best_match_idx],
                known_face_names[best_match_idx],
                confidence,
                "Face recognized"
            )
    
    return None, None, 0.0, "Face not recognized"


def recognize_face_haar(image):
    """Fallback: Recognize face using Haar Cascade (limited)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None, None, 0.0, "No face detected"
    
    # Haar cascade can only detect, not recognize
    # This is a placeholder - real recognition needs face_recognition library
    return None, None, 0.0, "Face detected but recognition requires face_recognition library"


def enroll_face_dlib(image, user_id, name):
    """Enroll a new face using face_recognition library"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find faces
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return False, "No face detected in image"
    
    if len(face_locations) > 1:
        return False, "Multiple faces detected, please use image with single face"
    
    # Get face encoding
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if not face_encodings:
        return False, "Could not encode face"
    
    face_encoding = face_encodings[0]
    
    # Save to file
    data = {
        'user_id': user_id,
        'name': name,
        'encoding': face_encoding,
        'enrolled_at': datetime.now().isoformat()
    }
    
    filename = f"{user_id}_{name.replace(' ', '_')}.pkl"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    # Reload faces
    load_known_faces()
    
    return True, f"Face enrolled successfully for {name}"


# ==================== API Endpoints ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'service': 'face-recognition',
        'status': 'running',
        'face_recognition_lib': USE_FACE_RECOGNITION,
        'known_faces_count': len(known_face_names)
    })


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize face from image
    
    Request JSON:
    {
        "image": "base64_encoded_image"
    }
    
    Response JSON:
    {
        "success": true,
        "recognized": true/false,
        "user_id": 1,
        "name": "John",
        "confidence": 0.85,
        "message": "Face recognized"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'recognized': False,
                'error': 'No image provided'
            }), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({
                'success': False,
                'recognized': False,
                'error': 'Invalid image format'
            }), 400
        
        print(f"📸 Received image: {image.shape}")
        
        # Recognize face
        if USE_FACE_RECOGNITION:
            user_id, name, confidence, message = recognize_face_dlib(image)
        else:
            user_id, name, confidence, message = recognize_face_haar(image)
        
        recognized = user_id is not None
        
        print(f"{'✅' if recognized else '❌'} Recognition result: {message}")
        
        return jsonify({
            'success': True,
            'recognized': recognized,
            'user_id': user_id,
            'name': name,
            'confidence': round(confidence, 4) if confidence else 0,
            'message': message
        })
        
    except Exception as e:
        print(f"❌ Error in /recognize: {e}")
        return jsonify({
            'success': False,
            'recognized': False,
            'error': str(e)
        }), 500


@app.route('/enroll', methods=['POST'])
def enroll():
    """
    Enroll new face
    
    Request JSON:
    {
        "user_id": 1,
        "name": "John Doe",
        "image": "base64_encoded_image"
    }
    
    Response JSON:
    {
        "success": true,
        "message": "Face enrolled successfully"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        required_fields = ['user_id', 'name', 'image']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        user_id = data['user_id']
        name = data['name']
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image format'
            }), 400
        
        print(f"📝 Enrolling face for: {name} (ID: {user_id})")
        
        if not USE_FACE_RECOGNITION:
            return jsonify({
                'success': False,
                'error': 'Face enrollment requires face_recognition library. Please install it.'
            }), 500
        
        # Enroll face
        success, message = enroll_face_dlib(image, user_id, name)
        
        if success:
            print(f"✅ {message}")
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            print(f"❌ {message}")
            return jsonify({
                'success': False,
                'error': message
            }), 400
            
    except Exception as e:
        print(f"❌ Error in /enroll: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/reload', methods=['POST'])
def reload_faces():
    """Reload all known faces from disk"""
    try:
        count = load_known_faces()
        return jsonify({
            'success': True,
            'message': f'Reloaded {count} faces'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/faces', methods=['GET'])
def list_faces():
    """List all enrolled faces"""
    faces = []
    for i, name in enumerate(known_face_names):
        faces.append({
            'user_id': known_face_ids[i],
            'name': name
        })
    
    return jsonify({
        'success': True,
        'count': len(faces),
        'faces': faces
    })


@app.route('/faces/<int:user_id>', methods=['DELETE'])
def delete_face(user_id):
    """Delete enrolled face by user_id"""
    try:
        deleted = False
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.startswith(f"{user_id}_") and filename.endswith('.pkl'):
                filepath = os.path.join(KNOWN_FACES_DIR, filename)
                os.remove(filepath)
                deleted = True
                print(f"🗑️  Deleted: {filename}")
        
        if deleted:
            load_known_faces()  # Reload
            return jsonify({
                'success': True,
                'message': f'Face deleted for user_id: {user_id}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No face found for user_id: {user_id}'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== ESP32-CAM Streaming Test ====================

@app.route('/stream-test', methods=['GET'])
def stream_test_page():
    """Simple HTML page to test ESP32-CAM stream"""
    esp32_cam_url = request.args.get('url', 'http://192.168.1.58')
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>ESP32-CAM Test</title></head>
    <body>
        <h1>ESP32-CAM Stream Test</h1>
        <p>URL: {esp32_cam_url}</p>
        <img src="{esp32_cam_url}/cam-hi.jpg" style="max-width:640px" id="cam">
        <br><br>
        <button onclick="refresh()">Refresh</button>
        <button onclick="startAuto()">Auto Refresh</button>
        <button onclick="stopAuto()">Stop</button>
        <script>
            let interval;
            function refresh() {{
                document.getElementById('cam').src = '{esp32_cam_url}/cam-hi.jpg?' + Date.now();
            }}
            function startAuto() {{
                interval = setInterval(refresh, 500);
            }}
            function stopAuto() {{
                clearInterval(interval);
            }}
        </script>
    </body>
    </html>
    """


# ==================== Main ====================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏠 Smart Home Face Recognition Service")
    print("="*50)
    
    # Load known faces on startup
    print("\n📂 Loading known faces...")
    load_known_faces()
    
    print(f"\n🚀 Starting server on http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

