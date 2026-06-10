
import cv2
import time
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

from cam_main_helpers import (
    check_approachability,
    distance_point_to_line
)

# ==========================================================
# Configuration
# ==========================================================

CAMERA_NUMBER = 1
CONF_THRESHOLD = 0.7

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
    <title>YOLO Pose Stream</title>
</head>
<body>
    <h1>YOLO Pose Stream</h1>

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
            print("Failed to read camera stream.")
            time.sleep(0.01)
            continue

        frame_timestamp = time.time()

        results = model(
            frame,
            verbose=False,
            classes=[0]
        )[0]

        processed_data = []

        if results.keypoints is not None:

            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            kpts_list = results.keypoints.data.cpu().numpy()

            for index, kpts in enumerate(kpts_list):

                face_kpts = kpts[0:5]

                approachable = check_approachability(
                    face_kpts,
                    frame,
                    conf_thresh=CONF_THRESHOLD
                )

                face_serialized = [
                    [float(x), float(y), float(conf)]
                    for x, y, conf in face_kpts
                ]

                bbox = [float(x) for x in boxes[index]]

                processed_data.append({
                    "person_id": index,
                    "bbox": bbox,
                    "face_keypoints": face_serialized,
                    "approachable": approachable,
                    "timestamp": frame_timestamp
                })

                color = (
                    (0, 255, 0)
                    if approachable
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2
                )

                if approachable:
                    cv2.putText(
                        frame,
                        "APPROACHABLE",
                        (25, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.67,
                        (0, 255, 0),
                        2
                    )

                person_conf = confidences[index]

                cv2.putText(
                    frame,
                    f"Person {person_conf:.2f}",
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                for idx, pt in enumerate(face_serialized):

                    x, y, conf = pt

                    if conf < CONF_THRESHOLD:
                        continue

                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        4,
                        (255, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"{face_names[idx]}",
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"({x:.0f},{y:.0f})",
                        (int(x) + 5, int(y) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        frame,
                        f"{conf:.2f}",
                        (int(x) + 5, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (67, 255, 67),
                        1
                    )

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
