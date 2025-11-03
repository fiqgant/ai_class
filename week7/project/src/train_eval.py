# src/train_eval.py
import os, cv2, json, random
import numpy as np

# ========= KONFIGURASI =========
ROOT_DIR     = os.path.dirname(os.path.dirname(__file__))
DATA_DIR     = os.path.join(ROOT_DIR, "data")
RAW_DIR      = os.path.join(DATA_DIR, "raw")
MODEL_DIR    = os.path.join(DATA_DIR, "models")

FACE_SIZE    = (200, 200)

# Holdout
HOLDOUT_PCT  = 0.2     # 20% test per identitas
SHUFFLE_SEED = 42
# ===============================

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Enhance (CLAHE) untuk konsistensi train ↔ inference ----------
def enhance_face(gray_crop):
    """Tingkatkan kontras wajah (CLAHE) agar robust terhadap pencahayaan."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_crop)

# ---------- Dataset ----------
def _list_image_paths_per_label():
    per_label_paths = {}
    label2id, id2label = {}, {}
    cur = 0
    for name in sorted(os.listdir(RAW_DIR)):
        pdir = os.path.join(RAW_DIR, name)
        if not os.path.isdir(pdir):
            continue
        files = [os.path.join(pdir, f) for f in os.listdir(pdir)
                 if f.lower().endswith((".png",".jpg",".jpeg"))]
        if len(files) == 0:
            continue
        per_label_paths[name] = sorted(files)
        label2id[name] = cur
        id2label[cur] = name
        cur += 1
    return per_label_paths, label2id, id2label

def _split_holdout(per_label_paths, test_ratio=HOLDOUT_PCT, seed=SHUFFLE_SEED):
    rng = random.Random(seed)
    train, test = [], []
    for label, paths in per_label_paths.items():
        paths_copy = paths[:]
        rng.shuffle(paths_copy)
        n = len(paths_copy)
        n_test = max(1, int(round(test_ratio * n))) if n >= 3 else (1 if n >= 2 else 0)
        test_split = paths_copy[:n_test]
        train_split = paths_copy[n_test:]
        if len(train_split) == 0 and len(test_split) > 1:
            train_split.append(test_split.pop())
        train.extend([(p, label) for p in train_split])
        test.extend([(p, label) for p in test_split])
    return train, test

def _load_gray_resized(paths_and_labels, label2id, apply_enhance=True):
    X, y = [], []
    for p, lab in paths_and_labels:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, FACE_SIZE)
        if apply_enhance:
            img = enhance_face(img)
        X.append(img)
        y.append(label2id[lab])
    return X, np.array(y, dtype=np.int32)

# ---------- Metrik ----------
def _confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def _metrics_report(y_true, y_pred, id2label):
    acc = float((y_true == y_pred).sum()) / max(1, len(y_true))
    n_classes = len(id2label)
    cm = _confusion_matrix(y_true, y_pred, n_classes)
    per_class = {}
    for i in range(n_classes):
        true_i = (y_true == i)
        if true_i.sum() == 0:
            per_class[id2label[i]] = None
        else:
            per_class[id2label[i]] = float((y_pred[true_i] == i).sum()) / float(true_i.sum())
    return acc, cm, per_class

# ====== KALIBRASI: hitung ambang otomatis dari jarak training ======
def compute_calibration(rec_full, X_all, y_all, id2label, pctl=95, extra_margin=10.0):
    """
    Prediksi semua sampel training dengan model final, ambil jarak LBPH.
    Ambang per-kelas = persentil pctl + margin; Global = pctl(all) + margin.
    Nilai lebih longgar agar mengurangi 'unknown' pada kondisi real.
    """
    all_dists = []
    per_class = {id2label[i]: [] for i in range(len(id2label))}
    for img, yi in zip(X_all, y_all):
        _, dist = rec_full.predict(img)
        all_dists.append(float(dist))
        per_class[id2label[yi]].append(float(dist))

    thr_per_class = {}
    for name, dists in per_class.items():
        if len(dists) == 0:
            continue
        base = float(np.percentile(dists, pctl))
        thr_per_class[name] = base + float(extra_margin)

    global_thr = float(np.percentile(all_dists, pctl)) + float(extra_margin)
    return {
        "global_thr": global_thr,
        "thr_per_class": thr_per_class,
        "percentile": int(pctl),
        "extra_margin": float(extra_margin),
        "samples": {k: len(v) for k, v in per_class.items()}
    }

# ---------- Train → Eval (holdout) → Retrain full → Save ----------
def train_eval_save():
    per_label_paths, label2id, id2label = _list_image_paths_per_label()
    if len(per_label_paths) == 0:
        raise RuntimeError("Dataset kosong. Jalankan capture dulu (src/collect.py).")

    train_list, test_list = _split_holdout(per_label_paths, test_ratio=HOLDOUT_PCT, seed=SHUFFLE_SEED)
    X_tr, y_tr = _load_gray_resized(train_list, label2id, apply_enhance=True)
    X_te, y_te = _load_gray_resized(test_list,  label2id, apply_enhance=True)

    if len(X_tr) == 0 or len(X_te) == 0:
        raise RuntimeError("Split menghasilkan set kosong. Tambah data atau ubah HOLDOUT_PCT.")

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("cv2.face tidak tersedia. Pastikan 'opencv-contrib-python' terpasang.")

    print(f"[INFO] Train-set: {len(X_tr)} gambar, Test-set: {len(X_te)} gambar, Kelas: {len(label2id)}")

    # Train di train-set (baseline evaluasi)
    rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    rec.train(X_tr, y_tr)

    # Evaluasi di test-set
    preds = [rec.predict(img)[0] for img in X_te]
    y_pred = np.array(preds, dtype=np.int32)

    acc, cm, per_class = _metrics_report(y_te, y_pred, id2label)
    print("\n[EVAL — HOLDOUT]")
    print(f"- Accuracy total: {acc:.4f}")
    print("- Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("- Per-class accuracy:")
    for k, v in per_class.items():
        if v is None:
            print(f"  {k:>12}: N/A (tidak ada sampel di test)")
        else:
            print(f"  {k:>12}: {v:.4f}")

    # Retrain di seluruh data (train+test) → simpan
    print("\n[INFO] Retrain di seluruh data untuk model final…")
    all_list = []
    for lab, paths in per_label_paths.items():
        all_list.extend([(p, lab) for p in paths])
    X_all, y_all = _load_gray_resized(all_list, label2id, apply_enhance=True)

    rec_full = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    rec_full.train(X_all, y_all)

    # Kalibrasi otomatis (lebih toleran)
    calib = compute_calibration(rec_full, X_all, y_all, id2label, pctl=95, extra_margin=10.0)

    model_path  = os.path.join(MODEL_DIR, "lbph_model.yml")
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    calib_path  = os.path.join(MODEL_DIR, "calibration.json")

    rec_full.write(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    # Ringkasan kalibrasi
    print("\n[CALIBRATION]")
    print(f"- Mode : percentile={calib['percentile']} + margin={calib['extra_margin']}")
    print(f"- Global threshold : {calib['global_thr']:.2f}")
    print("- Per-class thresholds:")
    for name, thr in calib["thr_per_class"].items():
        print(f"  {name:>12}: {thr:.2f}  (n={calib['samples'].get(name, 0)})")

    print(f"\n[SAVED] Model      : {model_path}")
    print(f"[SAVED] Labels     : {labels_path}")
    print(f"[SAVED] Calibration: {calib_path}")
    print("[DONE] Training + Evaluasi + Kalibrasi selesai.")

def main():
    print("="*56)
    print("TRAIN LBPH → HOLDOUT EVAL → RETRAIN FULL → SIMPAN MODEL (+CALIBRATION)")
    print("="*56)
    train_eval_save()

if __name__ == "__main__":
    main()
