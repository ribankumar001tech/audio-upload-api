
## System Requirements

| Requirement | Minimum |
|-------------|---------|
| OS | Ubuntu 20.04+ / Debian 11+ |
| Python | 3.10+ |
| RAM | 8 GB (16 GB recommended) |
| GPU | NVIDIA L4 / T4 / A10 with 16+ GB VRAM (for Ollama) |
| Disk | 50 GB (models + audio storage) |
| ffmpeg | Required for audio conversion |

---

## External Services

| Service | Purpose | Required |
|---------|---------|----------|
| **Sarvam AI** | Primary transcription (Hindi/Hinglish with diarization) | Yes |

---

## Server Setup

### 1. Install System Dependencies

```bash
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip ffmpeg git
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 6. Configure Environment

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
# Flask
SECRET_KEY=your-secret-key-here

# Transcription - Sarvam AI (primary)
SARVAM_API_KEY=your-sarvam-api-key

```

### 7. Test Run

```bash
python3 app.py
```

Access at: `http://<server-ip>:3000`
