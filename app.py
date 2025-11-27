"""
Smart Home Face Recognition Service
Flask REST API for face recognition using face_recognition library
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import base64
import os
import json
from datetime import datetime
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for Golang backend

# Configuration
KNOWN_FACES_DIR = "known_faces"
CONFIDENCE_THRESHOLD = 0.6  # Lower = more strict (0.6 is recommended)

# Create known_faces directory if not exists
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load known faces on startup
known_face_encodings = []
known_face_names = []
known_face_user_ids = []


def load_known_faces():
    """Load all known faces from known_faces directory"""
    global known_face_encodings, known_face_names, known_face_user_ids
    
    known_face_encodings = []
    known_face_names = []
    known_face_user_ids = []
    
    metadata_file = os.path.join(KNOWN_FACES_DIR, "metadata.json")
    
    if not os.path.exists(metadata_file):
        print("  No metadata.json found. No faces loaded.")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    for user_data in metadata:
        user_id = user_data['user_id']
        name = user_data['name']
        image_path = user_data['image_path']
        
        full_path = os.path.join(KNOWN_FACES_DIR, image_path)
        
        if os.path.exists(full_path):
            # Load image and get face encoding
            image = face_recognition.load_image_file(full_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                known_face_user_ids.append(user_id)
                print(f" Loaded: {name} (user_id: {user_id})")
            else:
                print(f" No face found in {image_path}")
        else:
            print(f"  File not found: {full_path}")
    
    print(f"\n Total faces loaded: {len(known_face_encodings)}\n")


def base64_to_image(base64_string):
    """Convert base64 string to numpy array (image)"""
    try:
        # Remove data:image/jpeg;base64, prefix if exists
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB (OpenCV uses BGR, face_recognition uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    except Exception as e:
        print(f"❌ Error decoding base64: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Face Recognition Service",
        "known_faces": len(known_face_encodings)
    }), 200


@app.route('/recognize', methods=['POST'])
def recognize_face():
    """
    Recognize face from base64 image
    
    Request JSON:
    {
        "image": "base64_encoded_image_string"
    }
    
    Response:
    {
        "success": true,
        "recognized": true,
        "user_id": 123,
        "name": "John Doe",
        "confidence": 0.45,
        "message": "Face recognized"
    }
    """
    try:
        # Get base64 image from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        base64_image = data['image']
        
        # Convert base64 to image
        image = base64_to_image(base64_image)
        
        if image is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image"
            }), 400
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            return jsonify({
                "success": True,
                "recognized": False,
                "message": "No face detected in image"
            }), 200
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({
                "success": True,
                "recognized": False,
                "message": "Could not encode face"
            }), 200
        
        # Compare with known faces
        face_encoding = face_encodings[0]  # Take first face
        
        if len(known_face_encodings) == 0:
            return jsonify({
                "success": True,
                "recognized": False,
                "message": "No known faces in database"
            }), 200
        
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Check if match is good enough
        if best_distance < CONFIDENCE_THRESHOLD:
            # Face recognized!
            user_id = known_face_user_ids[best_match_index]
            name = known_face_names[best_match_index]
            confidence = 1 - best_distance  # Convert distance to confidence
            
            print(f" Recognized: {name} (confidence: {confidence:.2f})")
            
            return jsonify({
                "success": True,
                "recognized": True,
                "user_id": user_id,
                "name": name,
                "confidence": round(confidence, 2),
                "message": f"Face recognized: {name}"
            }), 200
        else:
            # Face not recognized
            print(f"❌ Unknown face (best distance: {best_distance:.2f})")
            
            return jsonify({
                "success": True,
                "recognized": False,
                "message": "Face not recognized",
                "best_distance": round(best_distance, 2)
            }), 200
    
    except Exception as e:
        print(f"❌ Error in recognize_face: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/enroll', methods=['POST'])
def enroll_face():
    """
    Enroll new face to database
    
    Request JSON:
    {
        "user_id": 123,
        "name": "John Doe",
        "image": "base64_encoded_image_string"
    }
    
    Response:
    {
        "success": true,
        "message": "Face enrolled successfully",
        "image_path": "user_123_timestamp.jpg"
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'user_id' not in data or 'name' not in data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required fields: user_id, name, image"
            }), 400
        
        user_id = data['user_id']
        name = data['name']
        base64_image = data['image']
        
        # Convert base64 to image
        image = base64_to_image(base64_image)
        
        if image is None:
            return jsonify({
                "success": False,
                "error": "Failed to decode image"
            }), 400
        
        # Check if face exists in image
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            return jsonify({
                "success": False,
                "error": "No face detected in image"
            }), 400
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({
                "success": False,
                "error": "Could not encode face"
            }), 400
        
        # Save image to known_faces directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_{user_id}_{timestamp}.jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        
        # Convert RGB back to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_bgr)
        
        # Update metadata.json
        metadata_file = os.path.join(KNOWN_FACES_DIR, "metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        # Add new entry
        metadata.append({
            "user_id": user_id,
            "name": name,
            "image_path": filename,
            "enrolled_at": datetime.now().isoformat()
        })
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Reload known faces
        load_known_faces()
        
        print(f" Enrolled: {name} (user_id: {user_id})")
        
        return jsonify({
            "success": True,
            "message": f"Face enrolled successfully: {name}",
            "image_path": filename
        }), 200
    
    except Exception as e:
        print(f"❌ Error in enroll_face: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/reload', methods=['POST'])
def reload_faces():
    """Reload all known faces from disk"""
    try:
        load_known_faces()
        return jsonify({
            "success": True,
            "message": "Faces reloaded successfully",
            "known_faces": len(known_face_encodings)
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/list', methods=['GET'])
def list_faces():
    """List all known faces"""
    faces = []
    for i in range(len(known_face_names)):
        faces.append({
            "user_id": known_face_user_ids[i],
            "name": known_face_names[i]
        })
    
    return jsonify({
        "success": True,
        "total": len(faces),
        "faces": faces
    }), 200


if __name__ == '__main__':
    print("=" * 50)
    print(" Smart Home Face Recognition Service")
    print("=" * 50)
    
    # Load known faces on startup
    load_known_faces()
    
    # Start Flask server
    print("\n Starting Flask server on http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
