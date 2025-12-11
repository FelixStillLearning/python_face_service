"""
Test Face Recognition - Compare webcam capture dengan enrolled PKL files
"""
import cv2
import base64
import requests
import json

PYTHON_SERVICE_URL = "http://localhost:5001"

def capture_from_webcam():
    """Capture foto dari webcam"""
    print("📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return None
    
    print("✅ Webcam opened. Press SPACE to capture, ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Webcam - Press SPACE to capture', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("❌ Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE
            print("📸 Photo captured!")
            cap.release()
            cv2.destroyAllWindows()
            return frame
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def image_to_base64(image):
    """Convert OpenCV image ke Base64"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def test_recognition(image_base64):
    """Test recognize face dengan Python service"""
    print("\n🔍 Testing face recognition...")
    
    try:
        response = requests.post(
            f"{PYTHON_SERVICE_URL}/recognize-base64",
            json={"image": image_base64},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get('recognized'):
            print(f"\n✅ FACE RECOGNIZED!")
            print(f"   User ID: {result.get('user_id')}")
            print(f"   Name: {result.get('name')}")
            print(f"   Confidence: {result.get('confidence')}")
            print(f"\n🚪 Door would UNLOCK for user: {result.get('name')}")
        else:
            print(f"\n❌ FACE NOT RECOGNIZED")
            print(f"   Message: {result.get('message')}")
            print(f"\n🔔 Buzzer would SOUND - Unknown person!")
        
        return result
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("="*60)
    print("🎭 FACE RECOGNITION TEST")
    print("="*60)
    print("\nInstruksi:")
    print("1. Pastikan Python face service jalan (port 5001)")
    print("2. Posisikan wajah Anda di depan webcam")
    print("3. Tekan SPACE untuk capture foto")
    print("4. Sistem akan cek apakah wajah Anda terdaftar\n")
    
    # Capture foto
    image = capture_from_webcam()
    if image is None:
        return
    
    # Convert ke Base64
    print("🔄 Converting image to Base64...")
    image_base64 = image_to_base64(image)
    print(f"✅ Image size: {len(image_base64)} characters")
    
    # Test recognition
    result = test_recognition(image_base64)
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    main()
