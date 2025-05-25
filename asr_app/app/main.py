from fastapi import FastAPI, File, UploadFile, HTTPException
from pydub import AudioSegment
import os
import tempfile

import onnxruntime
import numpy as np
import librosa
import soundfile as sf


with open("app/model/vocab.txt", "r", encoding="utf-8") as f:
    loaded_vocab = [line.strip() for line in f]


class ASRModel:
    def __init__(self, model_path="app/model/asr_model.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        # Add blank token (usually last index in CTC)
        self.labels = loaded_vocab+['']

    def preprocess(self, audio_path):
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            raise ValueError("Sample rate must be 16kHz")

        # Compute log Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=400,           # 25 ms window for 16kHz audio
            hop_length=160,      # 10 ms hop
            n_mels=80            # 80 Mel bands
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # Normalize per Mel band (feature dimension)
        mean = np.mean(log_mel_spectrogram, axis=1, keepdims=True)
        std = np.std(log_mel_spectrogram, axis=1, keepdims=True)
        norm_log_mel = (log_mel_spectrogram - mean) / (std + 1e-6)

        # Input shape expected: (batch=1, features=80, time)
        input_tensor = np.expand_dims(norm_log_mel, axis=0).astype(np.float32)
        return input_tensor


    def postprocess(self, logits):
        if logits.size == 0:
            raise ValueError("Empty logits received from model")

        pred_ids = np.argmax(logits, axis=-1)[0]
        blank_idx = len(self.labels) - 1

        # Collapse repeating tokens and remove blanks
        decoded = []
        prev = None
        for idx in pred_ids:
            if idx != blank_idx and idx != prev:
                decoded.append(self.labels[idx])
            prev = idx

        out = ''.join(decoded).replace('▁', " ")
        return out


    def infer(self, audio_path):
        input_data = self.preprocess(audio_path)
        length = np.array([input_data.shape[2]]).astype(np.int64)
        ort_inputs = {
            self.session.get_inputs()[0].name: input_data,
            self.session.get_inputs()[1].name: length
        }
        ort_outs = self.session.run(None, ort_inputs)
        return self.postprocess(ort_outs[0])
    







app = FastAPI()
model = ASRModel()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Use NamedTemporaryFile for automatic cleanup and safer temp handling
    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio_file:
            temp_audio_path = temp_audio_file.name
            temp_audio_file.write(await file.read())

        # Convert audio to 16kHz mono wav using pydub
        audio = AudioSegment.from_file(temp_audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            wav_path = temp_wav_file.name
            audio.export(wav_path, format="wav")

        # Optional: print model input names for debugging
        # print([inp.name for inp in model.session.get_inputs()])

        # Perform transcription
        transcription = model.infer(wav_path)

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp files safely if they exist
        for path in [temp_audio_path, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)
