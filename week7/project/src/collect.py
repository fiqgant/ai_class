import os, cv2, time
import numpy as np

# ========= KONFIGURASI =========
ROOT_DIR       = os.path.dirname(os.path.dirname(__file__))  # folder repo
DATA_DIR       = os.path.join(ROOT_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")

CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE      = (200, 200)     # ukuran crop wajah (grayscale)
MIN_PER_PERSON = 20             # saran minimum per identitas (lebih banyak lebih bagus)
AUTO_SAVE_EVERY= 3              # auto-simpan setiap N frame saat wajah terdeteksi
# ===============================

os.makedirs(RAW_DIR, exist_ok=True)
detector = cv2.CascadeClassifier(CASCADE_PATH)

def _pick_largest_face(faces):
    if isinstance(faces, np.ndarray) and len(faces) > 0:
        areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in faces]
        return max(areas, key=lambda t: t[0])[1]
    return None

def _open_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka webcam (index 0).")
    return cap

def capture_one_person(person_name: str):
    """Buka webcam, capture wajah untuk nama tertentu, simpan ke data/raw/<nama>."""
    person_dir = os.path.join(RAW_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    existing = len([f for f in os.listdir(person_dir)
                    if f.lower().endswith((".png",".jpg",".jpeg"))])

    cap = _open_webcam()
    print(f"\n[CAPTURE] '{person_name}' — tekan 's' untuk simpan, 'q' untuk selesai orang ini.")
    saved, auto_ctr = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] frame webcam gagal, coba lagi...")
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))
        box = _pick_largest_face(faces)
        display = frame.copy()
        face_gray = None

        if box is not None:
            x,y,w,h = box
            pad = int(0.15 * max(w,h))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad); y1 = min(frame.shape[0], y + h + pad)
            cv2.rectangle(display, (x0,y0), (x1,y1), (0,255,0), 2)
            face_gray = gray[y0:y1, x0:x1]
            if face_gray.size > 0:
                face_gray = cv2.resize(face_gray, FACE_SIZE)

            auto_ctr += 1
            if face_gray is not None and auto_ctr % AUTO_SAVE_EVERY == 0:
                fname = f"{int(time.time())}_{existing+saved:04d}.png"
                cv2.imwrite(os.path.join(person_dir, fname), face_gray)
                saved += 1

        cv2.putText(display, f"Nama: {person_name} | Tersimpan: {existing+saved}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Capture Wajah", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and face_gray is not None:
            fname = f"{int(time.time())}_{existing+saved:04d}.png"
            cv2.imwrite(os.path.join(person_dir, fname), face_gray)
            saved += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    total = existing + saved
    print(f"[DONE] '{person_name}': total gambar = {total}")
    if total < MIN_PER_PERSON:
        print(f"[NOTE] Disarankan ≥ {MIN_PER_PERSON} gambar. Tambah lagi untuk akurasi lebih baik.")
    return total

def main():
    print("="*58)
    print("CAPTURE DATASET WAJAH VIA WEBCAM (per identitas)")
    print("Masukkan nama orang. ENTER kosong → selesai.")
    print("="*58)

    while True:
        name = input("Nama orang (ENTER kosong untuk selesai): ").strip().lower()
        if name == "":
            break
        capture_one_person(name)

    print("\n[INFO] Selesai capture. Lanjutkan ke training dengan 'python src/train_eval.py'")

if __name__ == "__main__":
    main()
