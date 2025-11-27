# 📋 Setup Instructions - Python Face Service Repository

## 🎯 Cara Membuat Repository Terpisah untuk Python Face Service

### Step 1: Inisialisasi Git Repository

```cmd
cd d:\Development\Projects\Active\Pemograman Iot\Smarthome\python_face_service
git init
```

### Step 2: Add All Files

```cmd
git add .
git status
```

Pastikan file berikut **TIDAK** ter-commit (sudah di `.gitignore`):
- ❌ `known_faces/*.jpg` (foto wajah user - privacy!)
- ❌ `venv/` (virtual environment)
- ❌ `__pycache__/` (compiled Python)

### Step 3: First Commit

```cmd
git config user.name "Kysu-dev"
git config user.email "your-email@example.com"

git commit -m "Initial commit: Face Recognition Service for Smart Home IoT"
```

### Step 4: Create GitHub Repository

1. Buka https://github.com/new
2. Repository name: `smart-home-face-recognition`
3. Description: `Python Flask face recognition service for Smart Home IoT - AI-powered access control`
4. **Public** atau **Private** (⚠️ Recommended: **Private** karena face data sensitif)
5. ❌ **JANGAN** centang "Initialize with README"
6. Click **Create repository**

### Step 5: Push ke GitHub

```cmd
git remote add origin https://github.com/Kysu-dev/smart-home-face-recognition.git
git branch -M main
git push -u origin main
```

### Step 6: Verify

Buka: `https://github.com/Kysu-dev/smart-home-face-recognition`

---

## 🔄 Update Repository (Setelah Edit Code)

```cmd
cd d:\Development\Projects\Active\Pemograman Iot\Smarthome\python_face_service

# Cek perubahan
git status

# Add perubahan
git add .

# Commit dengan pesan
git commit -m "Update: improve face recognition accuracy with CNN model"

# Push ke GitHub
git push
```

---

## 📁 Struktur Repository yang Akan Dibuat

```
smart-home-face-recognition/
├── .gitignore              # ✅ Existing (updated)
├── README.md              # ✅ Existing (updated)
├── SETUP.md               # ✅ This file
├── app.py                 # ✅ Existing
├── requirements.txt       # ✅ Existing
├── test_service.py       # ✅ Existing
└── known_faces/           # ✅ Existing
    └── .gitkeep          # ✅ Keep folder structure
```

---

## 🔐 Privacy & Security Notes

### ⚠️ PENTING: Face Data Privacy

**Known faces folder** berisi foto wajah user yang **TIDAK BOLEH** di-commit ke GitHub!

Sudah di-handle oleh `.gitignore`:
```gitignore
known_faces/*.jpg
known_faces/*.jpeg
known_faces/*.png
!known_faces/.gitkeep
```

### Setup Known Faces di Environment Lain

Setelah clone repository:

```cmd
cd python_face_service

# Folder known_faces sudah ada (empty)
# Manual copy face images dari backup/source lain
copy C:\backup\faces\*.jpg known_faces\

# Atau download dari secure storage
# - Google Drive (private)
# - Encrypted USB
# - Local backup
```

### Backup Face Data (Best Practice)

```cmd
# Create encrypted backup
# Option 1: Zip dengan password
"C:\Program Files\7-Zip\7z.exe" a -tzip -pYOUR_PASSWORD known_faces_backup.zip known_faces\

# Option 2: Copy ke private storage
xcopy known_faces\ "D:\Private_Backup\smart_home_faces\" /E /I
```

---

## 🔗 Link Repository Lain

### Backend Go Repository
```
https://github.com/Kysu-dev/backend_GO_iot
```

### ESP32 IoT Repository
```
https://github.com/Kysu-dev/smart-home-esp32-iot
```

### Flutter App Repository
```
https://github.com/Kysu-dev/smart-home-flutter-app
```

---

## 📦 Dependencies Management

### Update Requirements

Setelah install package baru:

```cmd
# Activate venv
venv\Scripts\activate

# Install new package
pip install new-package

# Update requirements.txt
pip freeze > requirements.txt

# Commit changes
git add requirements.txt
git commit -m "Add new-package dependency"
git push
```

### Install Dependencies di Environment Baru

```cmd
# Clone repository
git clone https://github.com/Kysu-dev/smart-home-face-recognition.git
cd smart-home-face-recognition

# Create venv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Testing Before Push

Sebelum push ke GitHub, pastikan:

```cmd
# 1. Test service
python app.py
# Buka: http://localhost:5001

# 2. Test dengan curl
curl http://localhost:5001/

# 3. Test recognition (jika ada sample image)
python test_service.py

# 4. Check no credentials in code
git diff

# 5. Verify gitignore works
git status
# Pastikan known_faces/*.jpg TIDAK muncul
```

---

## 📝 Commit Message Guidelines

### Format
```
<type>: <subject>

<body (optional)>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code formatting
- `refactor`: Code restructuring
- `test`: Add/update tests
- `chore`: Maintenance tasks

### Examples

```cmd
git commit -m "feat: add CNN model support for better accuracy"

git commit -m "fix: handle no face detected error gracefully"

git commit -m "docs: update API endpoint documentation"

git commit -m "refactor: optimize face encoding performance"
```

---

## 🚀 Quick Commands Reference

```cmd
# Clone repository
git clone https://github.com/Kysu-dev/smart-home-face-recognition.git

# Setup environment
cd smart-home-face-recognition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run service
python app.py

# Update code
git pull
git add .
git commit -m "your message"
git push

# Check logs
git log --oneline --graph

# Create feature branch
git checkout -b feature/improve-accuracy
```

---

## 🔧 Environment Variables (Optional)

Untuk konfigurasi yang lebih flexible, buat `.env`:

```env
# .env (jangan commit file ini!)
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
FLASK_DEBUG=True
KNOWN_FACES_DIR=known_faces
TOLERANCE=0.6
BACKEND_URL=http://localhost:8080
```

Install python-dotenv:
```cmd
pip install python-dotenv
```

Load di `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()

HOST = os.getenv('FLASK_HOST', '0.0.0.0')
PORT = int(os.getenv('FLASK_PORT', 5001))
```

---

## ⚠️ Important Reminders

### ❌ JANGAN Commit:
- Face images (`known_faces/*.jpg`)
- Virtual environment (`venv/`)
- Environment variables (`.env`)
- API keys / passwords
- Large model files (> 100MB)

### ✅ HARUS Commit:
- Source code (`app.py`, `test_service.py`)
- Dependencies (`requirements.txt`)
- Documentation (`README.md`, `SETUP.md`)
- Configuration templates (`.env.example`)
- Empty folder markers (`.gitkeep`)

---

## 📞 Support & Documentation

- **Main Documentation**: `README.md`
- **API Docs**: Available via `/` endpoint
- **Issues**: GitHub Issues
- **Contact**: [GitHub Profile](https://github.com/Kysu-dev)

---

## 🎓 Learning Resources

### Face Recognition
- [face_recognition library docs](https://github.com/ageitgey/face_recognition)
- [dlib documentation](http://dlib.net/)

### Flask
- [Flask documentation](https://flask.palletsprojects.com/)
- [Flask RESTful APIs](https://flask-restful.readthedocs.io/)

---

**Created with ❤️ for Smart Home IoT Face Recognition**
