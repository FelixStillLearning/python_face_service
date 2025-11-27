"""
Test script for Face Recognition Service
Run this after starting app.py to verify the service is working
"""

import requests
import base64
import json

# Configuration
SERVICE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n1️⃣  Testing Health Endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_list_faces():
    """Test list faces endpoint"""
    print("\n2️⃣  Testing List Faces Endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/list")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_recognize_no_image():
    """Test recognize without image (should fail)"""
    print("\n3️⃣  Testing Recognize Without Image (should fail)...")
    try:
        response = requests.post(
            f"{SERVICE_URL}/recognize",
            json={}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 400  # Should return error
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def encode_image_to_base64(image_path):
    """Helper: Encode image file to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"   ❌ Error encoding image: {e}")
        return None

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 Testing Face Recognition Service")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Health Check
    if test_health():
        tests_passed += 1
    
    # Test 2: List Faces
    if test_list_faces():
        tests_passed += 1
    
    # Test 3: Recognize without image
    if test_recognize_no_image():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Tests Passed: {tests_passed}/{tests_total}")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("✅ All tests passed! Service is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the service logs.")
    
    print("\n💡 To test with actual face recognition:")
    print("   1. Enroll a face first (use Postman or curl)")
    print("   2. Capture image from ESP32-CAM")
    print("   3. Send to /recognize endpoint")

if __name__ == "__main__":
    main()
