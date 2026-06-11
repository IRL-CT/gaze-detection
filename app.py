
import cv2
import time
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

from cam_main_helpers import (
    process_frame
)

# ==========================================================
# Configuration
# ==========================================================

CAMERA_NUMBER = 0
CONF_THRESHOLD = 0.7
IS_360 = True

model = YOLO("yolo26n-pose.pt")

cap = cv2.VideoCapture(CAMERA_NUMBER)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

face_names = [
    "Nose",
    "L Eye",
    "R Eye",
    "L Ear",
    "R Ear"
]

# ==========================================================
# Flask App
# ==========================================================

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Gaze Detector Stream</title>
</head>
<body>
    <h1>Gaze Detector Stream</h1>

    <img src="/video_feed" width="960">

</body>
</html>
"""

# ==========================================================
# Frame Generator
# ==========================================================

def generate_frames():

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Failed to read camera stream: check CAMERA_NUMBER.")
            time.sleep(0.01)
            continue

        if IS_360:
            h, w = frame.shape[:2]

            top_frame = frame[:h//2, :]
            bottom_frame = frame[h//2:, :]

            top_results = model(top_frame, verbose=False, classes=[0])[0]
            bottom_results= model(bottom_frame, verbose=False, classes=[0])[0]
        else:
            results = model(frame, verbose=False, classes=[0])[0]

        frame_timestamp = time.time()

        if IS_360:
            process_frame(top_results, top_frame, conf_thresh=CONF_THRESHOLD)
            process_frame(bottom_results, bottom_frame, conf_thresh=CONF_THRESHOLD)
            frame = np.vstack((top_frame, bottom_frame))
        else:
            process_frame(results, frame, conf_thresh=CONF_THRESHOLD)

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )

# ==========================================================
# Routes
# ==========================================================

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
