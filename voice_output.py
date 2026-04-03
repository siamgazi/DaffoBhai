"""
parrot.py — Listens to you, repeats it back via TTS.

Requirements:
    pip install sounddevice numpy faster-whisper requests playsound3

Run:
    python parrot.py
"""

import os
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
import requests
from faster_whisper import WhisperModel
from playsound3 import playsound

# ── CONFIG ───────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
SILENCE_DURATION = 0.7
MIN_SPEECH_SECS  = 0.3
MAX_SPEECH_SECS  = 20.0
CHUNK_SECS       = 0.1
BLOCK_SIZE       = int(SAMPLE_RATE * CHUNK_SECS)
CALIBRATION_SECS = 2.0
THRESHOLD_MULT   = 4.0
MODEL_SIZE       = "tiny"

TTS_URL          = "http://localhost:5000"
TTS_SPEAKER_ID   = 14

# ── QUEUES ───────────────────────────────────────────────────────────────────
raw_queue        = queue.Queue()
transcribe_queue = queue.Queue()

def mic_callback(indata, frames, time, status):
    raw_queue.put(indata.copy().flatten())

# ── LOAD & PRE-WARM MODEL ────────────────────────────────────────────────────
print(f"Loading Whisper {MODEL_SIZE}...")
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=min(8, os.cpu_count() or 4),
    num_workers=2,
)
print("Warming up model...")
list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")[0])
print("Ready.\n")

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    """Send text to local Piper TTS server and play the result."""
    try:
        response = requests.post(TTS_URL, json={"text": text, "speaker_id": TTS_SPEAKER_ID}, timeout=10)
        if response.status_code == 200:
            # Write to a temp file so playsound can play it
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(response.content)
                tmp_path = f.name
            playsound(tmp_path)
            os.remove(tmp_path)
        else:
            print(f"[TTS error {response.status_code}]")
    except requests.exceptions.ConnectionError:
        print("[TTS server not reachable — is Piper running on port 5000?]")

# ── TRANSCRIPTION + SPEAK WORKER ─────────────────────────────────────────────
def transcription_worker():
    while True:
        audio = transcribe_queue.get()
        if audio is None:
            break
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        if text:
            print(f"You said: {text}")
            speak(text)

# ── CALIBRATE ────────────────────────────────────────────────────────────────
def calibrate() -> float:
    print(f"Calibrating — stay quiet for {CALIBRATION_SECS:.0f}s...")
    n_frames = int(CALIBRATION_SECS / CHUNK_SECS)
    rms_values = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=BLOCK_SIZE, callback=mic_callback):
        for _ in range(n_frames):
            try:
                frame = raw_queue.get(timeout=2)
                rms_values.append(float(np.sqrt(np.mean(frame ** 2))))
            except queue.Empty:
                pass
    while not raw_queue.empty():
        raw_queue.get_nowait()
    noise_floor = float(np.mean(rms_values)) if rms_values else 0.003
    threshold = noise_floor * THRESHOLD_MULT
    print(f"Noise floor: {noise_floor:.5f}  →  threshold: {threshold:.5f}\n")
    return threshold

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    threshold = calibrate()

    silence_frames    = int(SILENCE_DURATION / CHUNK_SECS)
    min_speech_frames = int(MIN_SPEECH_SECS  / CHUNK_SECS)
    max_speech_frames = int(MAX_SPEECH_SECS  / CHUNK_SECS)

    worker = threading.Thread(target=transcription_worker, daemon=True)
    worker.start()

    buffer       = []
    silent_count = 0
    recording    = False

    print("[READY] Speak — I'll repeat everything back. Ctrl-C to quit.\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=BLOCK_SIZE, callback=mic_callback):
        while True:
            try:
                frame = raw_queue.get(timeout=1)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(frame ** 2)))

            if rms > threshold:
                if not recording:
                    print("🎙 ...", end="\r")
                    recording = True
                buffer.append(frame)
                silent_count = 0

            elif recording:
                buffer.append(frame)
                silent_count += 1

                if silent_count >= silence_frames or len(buffer) >= max_speech_frames:
                    recording = False
                    print("     ", end="\r")

                    if len(buffer) >= min_speech_frames:
                        audio = np.concatenate(buffer)
                        transcribe_queue.put(audio)

                    buffer.clear()
                    silent_count = 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        transcribe_queue.put(None)
        print("\n[DONE]")