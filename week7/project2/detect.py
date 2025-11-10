# detect.py
from ultralytics import YOLO
import cv2
import time

# =======================
# Konfigurasi
# =======================
USE_WEBCAM = True            # True = pakai webcam index 0
VIDEO_SOURCE = "sample.mp4"  # dipakai kalau USE_WEBCAM = False
MODEL_WEIGHTS = "yolov8n.pt" # pretrained COCO, ringan & cepat


def draw_detections(frame_bgr, results, class_names):
    """
    frame_bgr: np.ndarray BGR
    results : ultralytics.engine.results.Results
    class_names: dict index->class_name
    """
    boxes = results.boxes
    if boxes is None:
        return frame_bgr

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    for box, score, class_id in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = box.astype(int)
        name = class_names[int(class_id)]
        label_text = f"{name} {score*100:.1f}%"

        # box hijau
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # background label
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame_bgr,
            (x1, y1 - th - baseline),
            (x1 + tw, y1),
            (0, 255, 0),
            thickness=-1,
        )

        # teks hitam di atas bg hijau
        cv2.putText(
            frame_bgr,
            label_text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame_bgr


def main():
    # 1. load model YOLO pretrained
    model = YOLO(MODEL_WEIGHTS)
    class_names = model.names  # dict {0:'person',1:'bicycle',...}

    # 2. buka sumber video
    if USE_WEBCAM:
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("ERROR: Tidak bisa buka sumber video.")
        return

    prev_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Selesai / tidak ada frame.")
            break

        # optional resize biar FPS naik
        # frame_bgr = cv2.resize(frame_bgr, (640, 360))

        # 3. inference YOLO
        results_list = model(frame_bgr, verbose=False)
        results = results_list[0]

        # 4. gambar bbox
        annotated = draw_detections(frame_bgr.copy(), results, class_names)

        # (Optional) hitung FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 5. tampilkan
        cv2.imshow("YOLOv8 Realtime Detection", annotated)

        # keluar dengan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
