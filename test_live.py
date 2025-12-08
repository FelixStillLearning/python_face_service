"""
LIVE FACE RECOGNITION TEST
==========================
Real-time preview dengan OpenCV window
- Lihat preview kamera langsung
- Press SPACE untuk enroll (daftar muka)
- Press R untuk recognize (kenali muka)
- Press Q untuk quit
"""

import cv2
import numpy as np
import requests
import time
import pickle
import os

# Konfigurasi
ESP32_CAM_IP = '10.124.88.102'
RESOLUTION = '640x480'
KNOWN_FACES_DIR = './known_faces'

# Load face_recognition jika ada
try:
    import face_recognition
    USE_DLIB = True
    print("✅ Using dlib face_recognition")
except:
    USE_DLIB = False
    print("⚠️  Using Haar Cascade (fallback)")
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known faces
known_encodings = []
known_names = []
known_ids = []

def load_known_faces():
    """Load semua muka yang sudah terdaftar"""
    global known_encodings, known_names, known_ids
    
    known_encodings = []
    known_names = []
    known_ids = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.pkl'):
            with open(os.path.join(KNOWN_FACES_DIR, filename), 'rb') as f:
                data = pickle.load(f)
                known_encodings.append(data['encoding'])
                known_names.append(data['name'])
                known_ids.append(data['user_id'])
    
    print(f"📂 Loaded {len(known_names)} faces: {known_names}")

def fetch_frame():
    """Ambil 1 frame dari ESP32-CAM"""
    url = f"http://{ESP32_CAM_IP}/{RESOLUTION}.jpg"
    
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        return None
    except:
        return None

def detect_and_draw_faces(frame):
    """Deteksi muka dan gambar kotak"""
    if USE_DLIB:
        # Pakai face_recognition (dlib)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        
        for (top, right, bottom, left) in face_locations:
            # Gambar kotak hijau
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "FACE DETECTED", (left, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, len(face_locations)
    else:
        # Pakai Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "FACE DETECTED", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, len(faces)

def enroll_face(frame, user_id, name):
    """Daftarkan muka baru"""
    if not USE_DLIB:
        print("❌ Enrollment butuh face_recognition library!")
        return False
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    
    if len(face_locations) == 0:
        print("❌ Tidak ada muka terdeteksi!")
        return False
    
    if len(face_locations) > 1:
        print("❌ Lebih dari 1 muka terdeteksi! Pastikan cuma 1 orang.")
        return False
    
    encodings = face_recognition.face_encodings(rgb, face_locations)
    
    if not encodings:
        print("❌ Gagal encode muka!")
        return False
    
    # Simpan
    data = {
        'user_id': user_id,
        'name': name,
        'encoding': encodings[0],
        'enrolled_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    filename = f"{user_id}_{name.replace(' ', '_')}.pkl"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Berhasil daftar: {name} (ID: {user_id})")
    load_known_faces()  # Reload
    return True

def recognize_face(frame):
    """Kenali muka dari frame"""
    if not USE_DLIB:
        print("❌ Recognition butuh face_recognition library!")
        return None, None, 0
    
    if len(known_encodings) == 0:
        print("❌ Belum ada muka terdaftar!")
        return None, None, 0
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    
    if len(face_locations) == 0:
        print("❌ Tidak ada muka terdeteksi!")
        return None, None, 0
    
    encodings = face_recognition.face_encodings(rgb, face_locations)
    
    if not encodings:
        return None, None, 0
    
    # Compare dengan semua muka terdaftar
    for encoding in encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = np.argmin(distances)
        confidence = 1 - distances[best_idx]
        
        if confidence >= 0.6:  # Threshold
            return known_ids[best_idx], known_names[best_idx], confidence
    
    return None, None, 0

def countdown_enroll(frame, seconds=4):
    """Countdown sebelum daftar dengan preview"""
    for i in range(seconds, 0, -1):
        # Fetch frame baru
        new_frame = fetch_frame()
        if new_frame is not None:
            frame = new_frame
        
        # Deteksi muka
        display_frame, face_count = detect_and_draw_faces(frame.copy())
        
        # Tulis countdown
        text = f"ENROLLING IN {i}..."
        cv2.putText(display_frame, text, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        if face_count == 0:
            cv2.putText(display_frame, "NO FACE!", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif face_count > 1:
            cv2.putText(display_frame, "MULTIPLE FACES!", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "READY!", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('ESP32-CAM Face Recognition', display_frame)
        cv2.waitKey(1000)  # 1 detik
    
    return frame

def main():
    """Main loop"""
    print("\n" + "="*60)
    print("🎥 LIVE FACE RECOGNITION TEST")
    print("="*60)
    print(f"📷 Camera: {ESP32_CAM_IP}")
    print(f"📐 Resolution: {RESOLUTION}")
    print("\n⌨️  Controls:")
    print("   SPACE  = Enroll (daftar muka)")
    print("   R      = Recognize (kenali muka)")
    print("   Q      = Quit")
    print("="*60 + "\n")
    
    # Load known faces
    load_known_faces()
    
    # Buat window
    cv2.namedWindow('ESP32-CAM Face Recognition', cv2.WINDOW_NORMAL)
    
    last_message = ""
    message_time = 0
    
    while True:
        # Ambil frame
        frame = fetch_frame()
        
        if frame is None:
            # Tampilkan error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "CAMERA ERROR!", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_frame, f"Check: {ESP32_CAM_IP}", (150, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow('ESP32-CAM Face Recognition', error_frame)
        else:
            # Deteksi muka dan gambar kotak
            display_frame, face_count = detect_and_draw_faces(frame.copy())
            
            # Tulis info
            cv2.putText(display_frame, f"Faces: {face_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Known: {len(known_names)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Tulis message kalau ada
            if time.time() - message_time < 3:  # Show for 3 seconds
                cv2.putText(display_frame, last_message, (10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Tulis controls
            cv2.putText(display_frame, "SPACE=Enroll | R=Recognize | Q=Quit", 
                       (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('ESP32-CAM Face Recognition', display_frame)
        
        # Handle keyboard
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Bye!")
            break
        
        elif key == ord(' '):  # SPACE = Enroll
            print("\n📝 ENROLL MODE")
            user_id = input("   User ID: ")
            name = input("   Name: ")
            
            if user_id and name:
                print(f"\n⏱️  Get ready! Enrolling in 4 seconds...")
                final_frame = countdown_enroll(frame, seconds=4)
                
                if enroll_face(final_frame, int(user_id), name):
                    last_message = f"ENROLLED: {name}"
                    message_time = time.time()
                else:
                    last_message = "ENROLL FAILED!"
                    message_time = time.time()
        
        elif key == ord('r') or key == ord('R'):  # R = Recognize
            print("\n🔍 RECOGNIZING...")
            
            if frame is not None:
                user_id, name, confidence = recognize_face(frame)
                
                if user_id is not None:
                    print(f"✅ RECOGNIZED: {name} (ID: {user_id})")
                    print(f"   Confidence: {confidence:.2%}")
                    last_message = f"RECOGNIZED: {name} ({confidence:.0%})"
                    message_time = time.time()
                else:
                    print("❌ NOT RECOGNIZED")
                    last_message = "NOT RECOGNIZED!"
                    message_time = time.time()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
