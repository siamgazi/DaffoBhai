"""
assistant.py — Face-aware voice assistant with Pico the animated bot

Two windows:
  "You"  — your camera feed with face/attention detection
  "Pico" — animated character that reacts and speaks

Requirements:
    pip install opencv-python mediapipe sounddevice numpy faster-whisper requests playsound3

Ollama:
    ollama pull tinyllama && ollama serve
Piper TTS:
    keep your Piper server on port 5000
"""

import os, cv2, queue, threading, tempfile, time, math
import numpy as np
import sounddevice as sd
import mediapipe as mp
import requests
from faster_whisper import WhisperModel
from playsound3 import playsound

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE      = 16000
SILENCE_DURATION = 0.8
MIN_SPEECH_SECS  = 0.3
MAX_SPEECH_SECS  = 20.0
CHUNK_SECS       = 0.1
BLOCK_SIZE       = int(SAMPLE_RATE * CHUNK_SECS)
THRESHOLD_MULT   = 4.0

MIN_DIST_CM  = 25
MAX_DIST_CM  = 80
FOCAL_PX     = 600
REAL_IRIS_MM = 11.7
GAZE_YAW_LIMIT   = 25
GAZE_PITCH_LIMIT  = 25

OLLAMA_MODEL  = "tinyllama"
OLLAMA_URL    = "http://localhost:11434/api/chat"
SYSTEM_PROMPT = (
    "You are Pico, a voice assistant running on the user's computer. "
    "You exist only to answer questions and complete tasks for the user. Nothing else. "

    "Identity rules: "
    "1. You are an AI assistant. You have no body, no city, no weather, no feelings, no opinions, no personal life. "
    "2. Never pretend to be human or have human experiences. "
    "3. If asked how you are, what you think, or how you feel — redirect immediately: just say you're here to help and answer their actual question. "
    "4. Never say 'I think', 'I feel', 'In my opinion', 'I believe' — you don't have opinions. State facts. "

    "Response rules: "
    "Rules: Only answer what was asked. 1 sentence max. No filler words at the start or end. "
    "5. Answer only what was asked. Nothing more. "
    "6. 1 sentence per reply. 2 only if the answer genuinely requires it. "
    "7. Never start with filler: no 'Sure!', 'Great!', 'Of course!', 'Certainly!', 'Absolutely!', 'Happy to help!'. "
    "8. Never end with filler: no 'Hope that helps!', 'Let me know if you need more!', 'Feel free to ask!'. "
    "9. No follow-up questions. No suggestions unless the user asks for them. "
    "10. Never simulate what the user might say next. Stop after your reply. "

    "Topic rules: "
    "11. Do not engage in small talk, casual chat, or personal conversation. "
    "12. If the user tries to chat casually, respond with one short helpful line and stop. "
    "13. Do not discuss other AI models, compare yourself to them, or comment on your own capabilities. "
    "14. If a question is outside your knowledge, say so in one sentence — no guessing, no making things up. "

    "Tone rules: "
    "15. Be concise, calm, and neutral. Not cold, not overly cheerful. "
    "16. Sound natural when spoken aloud — avoid bullet points, lists, or markdown in your replies. "
    "17. Never repeat what the user just said back to them before answering. Just answer."
)

TTS_URL        = "http://localhost:5000"
TTS_SPEAKER_ID = 14

# ═══════════════════════════════════════════════════════════════════════════════
# STATES
# ═══════════════════════════════════════════════════════════════════════════════
IDLE       = "idle"
ATTENTIVE  = "attentive"
LISTENING  = "listening"
PROCESSING = "processing"
SPEAKING   = "speaking"

STATE_COLORS_BGR = {
    IDLE:       (80,  80,  80),
    ATTENTIVE:  (200, 200,  0),
    LISTENING:  (0,  220,   0),
    PROCESSING: (0,  165, 255),
    SPEAKING:   (220,  0, 200),
}
STATE_LABELS = {
    IDLE:       "Look at me motherjaat! (~50cm away)",
    ATTENTIVE:  "Hi! Speak to me :)",
    LISTENING:  "Listening...",
    PROCESSING: "khara vaiba loi...",
    SPEAKING:   "...",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════
_state      = IDLE
_state_lock = threading.Lock()
conversation_history = []
transcribe_queue: queue.Queue = queue.Queue()
raw_queue:        queue.Queue = queue.Queue()

def get_state():
    with _state_lock: return _state

def set_state(s):
    global _state
    with _state_lock: _state = s

# ═══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
POSE_IDXS  = [4, 152, 263, 33, 287, 57]
FACE_3D    = np.array([
    [  0.0,    0.0,    0.0], [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0], [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0], [ 150.0, -150.0, -125.0],
], dtype=np.float64)

def iris_distance_cm(lm, w, h):
    def diam(idxs):
        pts = np.array([(lm[i].x * w, lm[i].y * h) for i in idxs])
        return np.linalg.norm(pts[0] - pts[2]) * 2
    avg = (diam(LEFT_IRIS) + diam(RIGHT_IRIS)) / 2
    return None if avg < 1 else (FOCAL_PX * REAL_IRIS_MM) / avg / 10

def head_pose(lm, w, h):
    pts = np.array([[lm[i].x*w, lm[i].y*h] for i in POSE_IDXS], dtype=np.float64)
    cam = np.array([[FOCAL_PX,0,w/2],[0,FOCAL_PX,h/2],[0,0,1]], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(FACE_3D, pts, cam, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1]

def check_attention(lm, w, h):
    dist = iris_distance_cm(lm, w, h)
    if dist is None or not (MIN_DIST_CM <= dist <= MAX_DIST_CM): return False, dist
    try:
        pitch, yaw = head_pose(lm, w, h)
        return abs(yaw) < GAZE_YAW_LIMIT and abs(pitch) < GAZE_PITCH_LIMIT, dist
    except Exception:
        return True, dist

# ═══════════════════════════════════════════════════════════════════════════════
# PICO — CHARACTER RENDERER
# ═══════════════════════════════════════════════════════════════════════════════
# Palette
BG        = (18,  15,  30)    # deep indigo
HEAD      = (210, 235, 252)   # soft sky blue
HEAD_RIM  = (170, 205, 235)
EYE_WHITE = (252, 252, 252)
IRIS_COL  = (100, 145, 220)
PUPIL     = ( 28,  22,  45)
BROW_COL  = ( 90, 120, 190)
CHEEK     = (210, 175, 230)
MOUTH_COL = ( 80,  55, 110)
TEETH     = (248, 248, 252)
ANTENNA_S = (160, 195, 230)
ANTENNA_T = (255, 185, 100)
TEXT_COL  = (200, 200, 220)

W, H = 480, 520
CX, CY = W // 2, H // 2 + 20    # face centre
R  = 130                          # head radius

def _circle(img, c, r, color, t=-1):
    cv2.circle(img, c, r, color, t, cv2.LINE_AA)

def _ellipse(img, c, axes, a, s, e, color, t=-1):
    cv2.ellipse(img, c, axes, a, s, e, color, t, cv2.LINE_AA)

def draw_pico(state: str, t: float) -> np.ndarray:
    img = np.full((H, W, 3), BG, dtype=np.uint8)

    # ── blink ──
    blink = (t % 4.5) < 0.13

    # ── mouth openness (0‥1) ──
    if state == SPEAKING:
        mo = (math.sin(t * 10) + 1) / 2          # oscillates at ~5 Hz
    elif state == LISTENING:
        mo = 0.25 + 0.12 * math.sin(t * 3)        # gentle breathing
    else:
        mo = 0.0

    # ── eyebrow raise ──
    brow_lift = -10 if state in (PROCESSING, ATTENTIVE) else 0
    brow_lift += -6 * abs(math.sin(t * 0.4))       # very slow idle sway

    # ── subtle idle bob ──
    bob = int(5 * math.sin(t * 1.2))

    face_cy = CY + bob

    # ── antenna ──
    cv2.line(img, (CX, face_cy - R), (CX, face_cy - R - 38), ANTENNA_S, 4, cv2.LINE_AA)
    _circle(img, (CX, face_cy - R - 48), 12, ANTENNA_T)
    _circle(img, (CX, face_cy - R - 48), 12, (200, 145, 70), 2)
    _circle(img, (CX - 4, face_cy - R - 52), 4, (255, 230, 180))   # highlight

    # ── head shadow ──
    shadow = img.copy()
    _circle(shadow, (CX + 6, face_cy + 8), R, (8, 6, 18))
    cv2.addWeighted(shadow, 0.35, img, 0.65, 0, img)

    # ── head ──
    _circle(img, (CX, face_cy), R, HEAD)
    _circle(img, (CX, face_cy), R, HEAD_RIM, 3)

    # ── cheeks ──
    for side in [-1, 1]:
        ck = img.copy()
        _circle(ck, (CX + side * 78, face_cy + 22), 28, CHEEK)
        cv2.addWeighted(ck, 0.30, img, 0.70, 0, img)

    # ── eyes ──
    eye_y = face_cy - 22
    for side in [-1, 1]:
        ex = CX + side * 44

        # white
        ey_h = 4 if blink else 26
        _ellipse(img, (ex, eye_y), (22, ey_h), 0, 0, 360, EYE_WHITE)
        _ellipse(img, (ex, eye_y), (22, ey_h), 0, 0, 360, HEAD_RIM, 2)

        if not blink:
            # iris
            _circle(img, (ex, eye_y + 3), 13, IRIS_COL)
            # pupil
            _circle(img, (ex, eye_y + 3),  7, PUPIL)
            # highlight
            _circle(img, (ex - 5, eye_y - 2), 4, EYE_WHITE)
            _circle(img, (ex + 5, eye_y + 5), 2, (200, 220, 250))

    # ── eyebrows ──
    by = eye_y - 32 + brow_lift
    for side in [-1, 1]:
        bx0 = CX + side * 28
        bx1 = CX + side * 62
        by0 = int(by + (0 if side == 1 else 4))
        by1 = int(by + (4 if side == 1 else 0))
        cv2.line(img, (bx0, by0), (bx1, by1), BROW_COL, 3, cv2.LINE_AA)

    # ── mouth ──
    mouth_y = face_cy + 58
    if mo > 0.1:
        mw = 34
        mh = max(5, int(26 * mo))
        _ellipse(img, (CX, mouth_y), (mw, mh), 0, 0, 360, MOUTH_COL)
        # teeth strip
        _ellipse(img, (CX, mouth_y - mh // 3), (mw - 5, max(2, mh // 3)), 0, 0, 180, TEETH)
        _ellipse(img, (CX, mouth_y), (mw, mh), 0, 0, 360, MOUTH_COL, 2)
    else:
        # closed smile
        _ellipse(img, (CX, mouth_y - 4), (34, 14), 0, 0, 180, MOUTH_COL, 3)

    # ── state label ──
    label = STATE_LABELS[state]
    fs    = 0.52
    tw    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
    cv2.putText(img, label, ((W - tw) // 2, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, fs, TEXT_COL, 1, cv2.LINE_AA)

    # ── colour pulse ring around head ──
    ring_col = STATE_COLORS_BGR[state]
    alpha_r  = 0.45 + 0.25 * math.sin(t * 3)
    ring_img = img.copy()
    _circle(ring_img, (CX, face_cy), R + 8, ring_col, 3)
    cv2.addWeighted(ring_img, alpha_r, img, 1 - alpha_r, 0, img)

    return img

# ═══════════════════════════════════════════════════════════════════════════════
# WHISPER
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading Whisper tiny.en...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8",
                              cpu_threads=min(8, os.cpu_count() or 4), num_workers=2)
list(whisper_model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")[0])
print("Whisper ready.")

def transcribe(audio):
    segs, _ = whisper_model.transcribe(audio, language="en", beam_size=1, best_of=1,
                                        vad_filter=True, condition_on_previous_text=False,
                                        without_timestamps=True)
    return " ".join(s.text.strip() for s in segs if s.text.strip()) or None

# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA
# ═══════════════════════════════════════════════════════════════════════════════
def ask_ollama(text):
    conversation_history.append({"role": "user", "content": text})
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *conversation_history],
            "stream": False,
            "options": {
            "temperature": 0.1,
            "num_predict": 60,
            "num_thread": 8,
        },
        }, timeout=120)
        reply = r.json()["message"]["content"].strip()
    except Exception as e:
        reply = "Sorry, I had trouble thinking just now."
        print(f"[Ollama error] {e}")
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

# ═══════════════════════════════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════════════════════════════
def speak(text):
    try:
        r = requests.post(TTS_URL, json={"text": text, "speaker_id": TTS_SPEAKER_ID}, timeout=10)
        if r.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(r.content); path = f.name
            playsound(path); os.remove(path)
        else:
            print(f"[TTS error {r.status_code}]")
    except Exception as e:
        print(f"[TTS error] {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE WORKER
# ═══════════════════════════════════════════════════════════════════════════════
def pipeline_worker():
    while True:
        audio = transcribe_queue.get()
        if audio is None: break
        set_state(PROCESSING)
        text = transcribe(audio)
        if not text: set_state(ATTENTIVE); continue
        print(f"\nYou : {text}")
        reply = ask_ollama(text)
        print(f"Pico: {reply}\n")
        set_state(SPEAKING)
        speak(reply)
        set_state(ATTENTIVE)

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════════════════════════════════════════
def mic_callback(indata, frames, time, status):
    raw_queue.put(indata.copy().flatten())

def audio_thread(threshold):
    sf = int(SILENCE_DURATION / CHUNK_SECS)
    mn = int(MIN_SPEECH_SECS  / CHUNK_SECS)
    mx = int(MAX_SPEECH_SECS  / CHUNK_SECS)
    buf, sc, rec = [], 0, False
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=BLOCK_SIZE, callback=mic_callback):
        while True:
            try: frame = raw_queue.get(timeout=1)
            except queue.Empty: continue
            state = get_state()
            if state in (PROCESSING, SPEAKING, IDLE):
                buf.clear(); sc = 0; rec = False; continue
            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms > threshold:
                if not rec: set_state(LISTENING); rec = True
                buf.append(frame); sc = 0
            elif rec:
                buf.append(frame); sc += 1
                if sc >= sf or len(buf) >= mx:
                    rec = False
                    if len(buf) >= mn:
                        transcribe_queue.put(np.concatenate(buf))
                    buf.clear(); sc = 0

# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATE
# ═══════════════════════════════════════════════════════════════════════════════
def calibrate():
    print("Calibrating mic — stay quiet for 2s...")
    vals = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=BLOCK_SIZE, callback=mic_callback):
        for _ in range(int(2.0 / CHUNK_SECS)):
            try:
                frame = raw_queue.get(timeout=2)
                vals.append(float(np.sqrt(np.mean(frame ** 2))))
            except queue.Empty: pass
    while not raw_queue.empty(): raw_queue.get_nowait()
    nf = float(np.mean(vals)) if vals else 0.003
    t  = nf * THRESHOLD_MULT
    print(f"Mic threshold: {t:.5f}\n")
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA HUD
# ═══════════════════════════════════════════════════════════════════════════════
def draw_camera_hud(frame, state, dist_cm):
    h, w = frame.shape[:2]
    color = STATE_COLORS_BGR[state]
    cv2.rectangle(frame, (0, 0), (w, 44), (15, 15, 15), -1)
    cv2.circle(frame, (22, 22), 11, color, -1, cv2.LINE_AA)
    cv2.putText(frame, STATE_LABELS[state], (42, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if dist_cm is not None:
        dc = (0, 200, 80) if MIN_DIST_CM <= dist_cm <= MAX_DIST_CM else (0, 80, 220)
        cv2.putText(frame, f"{dist_cm:.0f}cm", (w - 90, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, dc, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 28), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, "q = quit", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    threshold = calibrate()

    threading.Thread(target=audio_thread,    args=(threshold,), daemon=True).start()
    threading.Thread(target=pipeline_worker,                     daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Position windows side by side
    cv2.namedWindow("You",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pico", cv2.WINDOW_NORMAL)
    cv2.moveWindow("You",    30,  80)
    cv2.moveWindow("Pico",  700,  80)
    cv2.resizeWindow("Pico", W, H)

    dist_cm = None
    start   = time.time()

    print("[READY] Look at Pico from ~50cm to start talking.\n")

    while True:
        ret, cam_frame = cap.read()
        if not ret: break
        cam_frame = cv2.flip(cam_frame, 1)
        fh, fw    = cam_frame.shape[:2]
        rgb       = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb)
        state     = get_state()

        if results.multi_face_landmarks:
            lm             = results.multi_face_landmarks[0].landmark
            attentive, dist_cm = check_attention(lm, fw, fh)

            if   state == IDLE      and attentive:  set_state(ATTENTIVE)
            elif state == ATTENTIVE and not attentive: set_state(IDLE)

            # dot overlay on face
            col = STATE_COLORS_BGR[get_state()]
            for idx in [33, 263, 1, 61, 291, 199, 4, 152]:
                p = lm[idx]
                cv2.circle(cam_frame, (int(p.x*fw), int(p.y*fh)), 2, col, -1)

        elif state in (ATTENTIVE, LISTENING):
            set_state(IDLE); dist_cm = None

        draw_camera_hud(cam_frame, get_state(), dist_cm)
        cv2.imshow("You", cam_frame)

        # ── Pico character ──
        t = time.time() - start
        pico_frame = draw_pico(get_state(), t)
        cv2.imshow("Pico", pico_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    transcribe_queue.put(None)
    print("\n[DONE]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        transcribe_queue.put(None)
        print("\n[DONE]")