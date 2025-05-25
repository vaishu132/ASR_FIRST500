---

# 🎙️ NeMo ASR FastAPI Server

Deploy a production-ready Automatic Speech Recognition (ASR) service using NVIDIA NeMo models with FastAPI, ONNX, and Docker. This project provides an easy-to-use REST API for transcribing speech in WAV audio files using a pre-trained `stt_hi_conformer_ctc_medium` model.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vaishu132/nemo-asr-fastapi.git
cd nemo-asr-fastapi
```

### 2. Build the Docker Image

```bash
docker build -t nemo-asr-api .
```

### 3. Run the Container

```bash
docker run -p 8000:8000 nemo-asr-api
```

The API will be accessible at `http://127.0.0.1:8000`.

---

## 🚀 Quick Start

Use `curl` or Postman to test the `/transcribe` endpoint:

### 🔁 Using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@path_to_your_audio.wav"
```

> Ensure your audio file is 16kHz mono WAV and between 5–10 seconds long.

---

## 🎯 Design Highlights

### ✅ Model Selection

I have used NVIDIA NeMo’s [stt\_hi\_conformer\_ctc\_medium](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium) for robust Hindi speech recognition.

### ⚡ ONNX Optimization

The original `.nemo` model is converted to ONNX format for accelerated inference with reduced latency.

## 📌 Important Note

please download the .onnx model from https://drive.google.com/file/d/1I2uQq0wHBy-  Jb3qWbKrAMwEF8IyHg5YS/view?usp=sharing and save it as app/model/asr_model.onnx
and then run the main.py

### 🧪 Audio Handling

* Ensures all input audio is:

  * PCM WAV
  * Sampled at 16kHz
  * Between 5–10 seconds
* Uses PyDub and NumPy for preprocessing

### 🚦 Endpoint Design

| Method | Endpoint      | Description                    |
| ------ | ------------- | ------------------------------ |
| POST   | `/transcribe` | Upload WAV file and transcribe |

---

## 🛠️ Project Structure

```
asr_app/
├── app/
│   ├── main.py               # FastAPI app with /transcribe endpoint
│   ├── _pycache_
│   ├── model/
│       └──vocab.txt
│       └── stt_hi_conformer.onnx # ONNX exported model (please download the .onnx model from https://drive.google.com/file/d/1I2uQq0wHBy-  Jb3qWbKrAMwEF8IyHg5YS/view?usp=sharing and save it as app/model/asr_model.onnx) 
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 🔍 Future Enhancements

* Batch transcription support
* Streaming inference via WebSocket
* Multilingual model support
* GPU inference via Triton Inference Server

 For long audios and file formats other than .wav

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Open an issue for bugs or feature requests
* Submit a PR for enhancements

---

---

## 🙋 Questions?

Feel free to open an issue if you need help or have suggestions!

---

