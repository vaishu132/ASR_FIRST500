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
### 2. Conversion to `.onnx`

I have used NVIDIA NeMo’s [stt\_hi\_conformer\_ctc\_medium](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium/files) and downloaded the `stt_hi_conformer_ctc_medium.nemo` for robust Hindi speech recognition.
The original `.nemo` model is converted to ONNX format for accelerated inference with reduced latency.
If you want to use any other model, please change the model path from `load_model_and_vocab.py` and then run it.

### 3. Build the Docker Image

```bash
docker build -t nemo-asr-api .
```

### 4. Run the Container

```bash
docker run -p 8000:8000 nemo-asr-api
```

The API will be accessible at `http://127.0.0.1:8000/docs`.

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

## 🛠️ Project Structure

```
asr_app/
├── app/
│   ├── main.py                 # FastAPI application with /transcribe endpoint          
│   └── model/
├── load_model_and_vocab.py 
├── Dockerfile
├── README.md
└── requirements.txt

```
---

## 📌 Important Note

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

## 🔍 Future Enhancements

* Batch transcription support – Enable processing of multiple audio files in a single request.
* Streaming inference via WebSocket – Support real-time transcription for streaming audio.
* Multilingual model support – Extend support to additional languages beyond the current model.
* GPU inference with Triton Inference Server – Integrate NVIDIA Triton for high-performance GPU-based inference.
* Support for longer audio files – Currently optimized for short clips (5–10 seconds); future versions will handle longer audio inputs.
* Additional audio format support – Currently supports `.wav` files; will add support for formats like `.mp3`.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Open an issue for bugs or feature requests
* Submit a PR for enhancements

---

## 🙋 Questions?

Feel free to open an issue if you need help or have suggestions!

---

