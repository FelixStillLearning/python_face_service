# 🐍 Smart Home Face Recognition Service

Python REST API untuk face recognition menggunakan library `face_recognition` (dlib-based).

## 🎯 Fungsi
Service ini menerima gambar base64 dari Golang backend, melakukan face recognition, dan mengembalikan hasil identifikasi user.

## 📡 API Endpoints

### 1. **Health Check**
```bash
GET http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "Face Recognition Service",
  "known_faces": 5
}
```

---

### 2. **Recognize Face** (Utama)
```bash
POST http://localhost:5000/recognize
Content-Type: application/json

{
  "image": "base64_encoded_image_string"
}
```

Response (Recognized):
```json
{
  "success": true,
  "recognized": true,
  "user_id": 123,
  "name": "John Doe",
  "confidence": 0.85,
  "message": "Face recognized: John Doe"
}
```

Response (Not Recognized):
```json
{
  "success": true,
  "recognized": false,
  "message": "Face not recognized"
}
```

---

### 3. **Enroll New Face**
```bash
POST http://localhost:5000/enroll
Content-Type: application/json

{
  "user_id": 123,
  "name": "John Doe",
  "image": "base64_encoded_image_string"
}
```

Response:
```json
{
  "success": true,
  "message": "Face enrolled successfully: John Doe",
  "image_path": "user_123_20231125_143022.jpg"
}
```

---

### 4. **Reload Faces**
```bash
POST http://localhost:5000/reload
```

---

### 5. **List Known Faces**
```bash
GET http://localhost:5000/list
```

---

## 📦 Installation

### Step 1: Install Python Dependencies
```bash
cd python_face_service
pip install -r requirements.txt
```

**⚠️ Catatan:** Library `dlib` (dependency dari `face_recognition`) memerlukan:
- **Windows:** Install Visual Studio Build Tools atau gunakan pre-built wheel
- **Linux:** `sudo apt-get install cmake`
- **macOS:** `brew install cmake`

---

## 🚀 Running the Service

```bash
python app.py
```

Service akan jalan di: **http://localhost:5000**

---

## 📁 File Structure

```
python_face_service/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── known_faces/          # Stored face images (auto-created)
    ├── metadata.json     # User metadata
    ├── user_1_timestamp.jpg
    ├── user_2_timestamp.jpg
    └── ...
```

---

## 🔄 Integration Flow

```
ESP32-CAM → Backend Golang → Python Service → Backend Golang → MQTT → ESP32
```

1. **ESP32-CAM** mengirim foto (base64) ke Backend Golang via HTTP
2. **Backend Golang** forward ke Python Service (POST /recognize)
3. **Python Service** proses face recognition, return user_id
4. **Backend Golang** simpan log, publish MQTT unlock command
5. **ESP32** terima MQTT, buka pintu

---

## ⚙️ Configuration

Edit di `app.py`:

```python
CONFIDENCE_THRESHOLD = 0.6  # Lower = more strict (0.4-0.6 recommended)
```

- **0.4**: Very strict (may reject valid faces)
- **0.6**: Recommended (balanced)
- **0.8**: Loose (may accept wrong faces)

---

## 🧪 Testing dengan cURL

### Test Health:
```bash
curl http://localhost:5000/health
```

### Test Recognize (dengan sample base64):
```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"your_base64_string_here\"}"
```

---

## 🔧 Troubleshooting

### Error: "No module named 'dlib'"
```bash
pip install dlib
# Atau di Windows:
pip install dlib-binary
```

### Error: "Could not import face_recognition"
```bash
pip uninstall face-recognition
pip install face-recognition
```

### Service tidak bisa connect dari Backend
- Pastikan Flask jalan di `0.0.0.0` (bukan `127.0.0.1`)
- Cek firewall Windows
- Verify dengan `curl http://localhost:5000/health`

---

## 📊 Performance

- **Recognize time:** ~200-500ms per image (tergantung jumlah known faces)
- **Memory usage:** ~200-500MB (tergantung jumlah enrolled faces)
- **Recommended:** Max 100 enrolled faces untuk performa optimal

---

## 🔐 Security Notes

- Service ini untuk **local network only** (tidak ada authentication)
- Jangan expose ke internet tanpa API key/auth
- Base64 images dapat besar (~100KB-1MB), sesuaikan timeout Backend Golang
