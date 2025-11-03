# app.py
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

MODEL_WEIGHTS = "yolov8n.pt"

# Load YOLO sekali di global scope biar nggak reload tiap frame
yolo_model = YOLO(MODEL_WEIGHTS)
CLASS_NAMES = yolo_model.names  # dict idx -> label


def draw_boxes_bgr(frame_bgr, results):
    boxes = results.boxes
    if boxes is None:
        return frame_bgr

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    for box, score, class_id in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = box.astype(int)
        class_id = int(class_id)
        label_text = f"{CLASS_NAMES[class_id]} {score*100:.1f}%"

        # kotak hijau
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # background label
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top_left = (x1, max(y1 - th - baseline, 0))
        bottom_right = (x1 + tw, y1)
        cv2.rectangle(frame_bgr, top_left, bottom_right, (0, 255, 0), -1)

        # text hitam
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


def detect_frame(frame_rgb):
    """
    frame_rgb: numpy array shape (H,W,3) RGB dari webcam browser.
    return: numpy array RGB sudah di-annotate bbox.
    """
    # convert RGB -> BGR buat OpenCV & YOLO
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # run YOLO sekali
    results_list = yolo_model(frame_bgr, verbose=False)
    results = results_list[0]

    # gambar bbox
    annotated_bgr = draw_boxes_bgr(frame_bgr.copy(), results)

    # balik lagi BGR -> RGB biar Gradio bisa tampilkan
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb


with gr.Blocks(title="YOLOv8 Realtime Webcam Detection") as demo:
    gr.Markdown(
        """
        # 🔍 YOLOv8 Realtime Detection via Webcam
        - Izinkan kamera di browser
        - Frame webcam dikirim terus menerus ke server
        - Server jalankan YOLOv8 & balikin hasil dengan bounding box
        """
    )

    with gr.Row():
        with gr.Column():
            cam_in = gr.Image(
                label="Webcam Input",
                sources=["webcam"],     # minta webcam browser
                streaming=True,         # kirim stream frame terus-menerus
                type="numpy",           # biar kita dapet np.ndarray RGB
            )
        with gr.Column():
            cam_out = gr.Image(
                label="Deteksi",
                type="numpy",
            )

    # stream() akan panggil detect_frame berkali-kali selama webcam aktif
    cam_in.stream(
        fn=detect_frame,
        inputs=cam_in,
        outputs=cam_out,
        time_limit=60,          # deteksi jalan 60 detik per sesi
        stream_every=0.1,       # deteksi ~10 fps (0.1s per call); adjust naik/turun
        concurrency_limit=1,    # satu user per proses (aman buat lokal)
    )

if __name__ == "__main__":
    demo.launch()
