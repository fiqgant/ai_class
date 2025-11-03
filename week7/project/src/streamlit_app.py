# src/streamlit_realtime.py
import os, json, cv2, numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# ====== PATH & KONFIG ======
ROOT_DIR     = os.path.dirname(os.path.dirname(__file__))     # repo root
MODEL_DIR    = os.path.join(ROOT_DIR, "data", "models")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE    = (200, 200)

# STUN server (supaya WebRTC bisa jalan di banyak jaringan)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_assets():
    model_path  = os.path.join(MODEL_DIR, "lbph_model.yml")
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    calib_path  = os.path.join(MODEL_DIR, "calibration.json")

    if not (os.path.exists(model_path) and os.path.exists(labels_path)):
        raise RuntimeError("Model/labels belum ada. Jalankan 'python src/train_eval.py' setelah 'python src/collect.py'.")

    if not (hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")):
        raise RuntimeError("cv2.face tidak tersedia. Instal 'opencv-contrib-python' (bukan opencv-python).")

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(model_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    id2label = {int(k): v for k, v in meta["id2label"].items()}

    calib = None
    if os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    return rec, id2label, detector, calib

def pick_largest(faces):
    if faces is None or len(faces) == 0:
        return None
    areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in faces]
    return max(areas, key=lambda t: t[0])[1]

def enhance_face(gray_crop):
    # CLAHE untuk bantu kontras pada pencahayaan buruk
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_crop)

def brightness_mean(gray):
    return float(np.mean(gray))

def sharpness_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def decide_threshold(label: str, calib: dict | None, relax: bool):
    # Prioritas: per-kelas > global > fallback
    relax_factor = 1.15 if relax else 1.0
    if calib:
        thr_per_class = calib.get("thr_per_class", {})
        if label in thr_per_class:
            return float(thr_per_class[label]) * relax_factor
        if "global_thr" in calib:
            return float(calib["global_thr"]) * relax_factor
    return 80.0 * relax_factor  # fallback konservatif

# ====== UI ======
st.set_page_config(page_title="Face ID (LBPH) — Realtime", layout="centered")
st.title("Face ID (LBPH) — Streamlit Realtime (Auto Threshold)")
st.caption("Live webcam dengan auto-threshold dari kalibrasi. Tidak perlu slider.")

# Opsi tampilan
col1, col2, col3, col4 = st.columns(4)
with col1:
    show_box = st.checkbox("Tampilkan bbox", value=True)
with col2:
    show_label = st.checkbox("Tampilkan label", value=True)
with col3:
    flip_view = st.checkbox("Mirror (selfie)", value=True)
with col4:
    relax_thr = st.checkbox("Lebih toleran", value=False, help="Naikkan threshold ~15% untuk kondisi sulit")

# Status info panel
info_box = st.empty()

# Load assets (model, labels, detector, calibration)
try:
    rec, id2label, detector, calib = load_assets()
    boot_err = None
except Exception as e:
    rec = id2label = detector = calib = None
    boot_err = str(e)

if boot_err:
    st.error(f"BOOT ERROR:\n{boot_err}")
    st.stop()

# ====== Video Transformer untuk streamlit-webrtc ======
class FaceIDTransformer:
    def __init__(self, rec, id2label, detector, calib):
        self.rec = rec
        self.id2label = id2label
        self.detector = detector
        self.calib = calib
        self.last_info = {"label": None, "distance": None, "thr_used": None, "ok": None,
                          "brightness": None, "sharpness": None}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Ambil frame BGR
        img_full = frame.to_ndarray(format="bgr24")

        # Downscale untuk deteksi (lebih cepat), simpan scale
        scale = 0.6  # deteksi di 60% resolusi
        small = cv2.resize(img_full, None, fx=scale, fy=scale)
        if flip_view:
            small = cv2.flip(small, 1)
            img_full = cv2.flip(img_full, 1)

        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # deteksi wajah (lebih permisif)
        faces = self.detector.detectMultiScale(gray_small, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))
        box = pick_largest(faces)

        out = img_full
        label_show = "no-face"
        dist_show = None
        ok = False
        thr_used = None
        b_mean = None
        sh_var = None

        if box is not None:
            # kembalikan box ke koordinat full-res
            x, y, w, h = [int(v / scale) for v in box]
            pad = int(0.10 * max(w, h))  # pad lebih ketat
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(img_full.shape[1], x + w + pad); y1 = min(img_full.shape[0], y + h + pad)
            face = cv2.cvtColor(img_full[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

            if face.size > 0:
                # kualitas
                b_mean = brightness_mean(face)
                sh_var = sharpness_laplacian(face)

                face = cv2.resize(face, FACE_SIZE)
                face_eq = enhance_face(face)

                pred_id, dist = self.rec.predict(face_eq)
                label = self.id2label.get(int(pred_id), "unknown")

                thr_used = decide_threshold(label, self.calib, relax=relax_thr)
                ok = (float(dist) <= thr_used)
                label_show = label if ok else "unknown"
                dist_show = float(dist)

                if show_box:
                    color = (0,255,0) if ok else (0,0,255)
                    cv2.rectangle(out, (x0,y0), (x1,y1), color, 2)

                if show_label:
                    conf = max(0.0, 1.0 - (float(dist) / (thr_used + 1e-6))) if thr_used else 0.0
                    txt = f"{label_show} | d={dist:.1f} | thr={thr_used:.1f} | conf={conf*100:.0f}%"
                    cv2.putText(out, txt, (x0, max(20,y0-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0,255,0) if ok else (0,0,255), 2)

                # overlay kualitas (kecil) di pojok bbox
                kval = []
                if b_mean is not None: kval.append(f"bright:{b_mean:.0f}")
                if sh_var is not None: kval.append(f"sharp:{sh_var:.0f}")
                if kval:
                    cv2.putText(out, " | ".join(kval), (x0, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

        self.last_info = {
            "label": label_show,
            "distance": dist_show,
            "thr_used": thr_used,
            "ok": ok,
            "brightness": b_mean,
            "sharpness": sh_var,
            "calibrated": self.calib is not None,
            "relax_thr": relax_thr
        }

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# Inisialisasi streamer
transformer = FaceIDTransformer(rec, id2label, detector, calib)

webrtc_ctx = webrtc_streamer(
    key="faceid-realtime",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=transformer.recv,
    media_stream_constraints={"video": True, "audio": False},
)

# Panel status yang update periodik
if webrtc_ctx.state.playing:
    st.markdown("**Status terakhir (≈1–2 detik):**")
    from time import sleep
    for _ in range(30):
        info_box.json(transformer.last_info)
        sleep(0.2)
