import cv2
import time
import numpy as np
# from flask import Flask, render_template
# from flask_socketio import SocketIO
from ultralytics import YOLO
from cam_main_helpers import process_frame

# Load real-time YOLO Pose model (Nano version recommended for speed)
model = YOLO('yolo26n-pose.pt')

# TODO: connect & test w/Insta360, 360 view
CAMERA_NUMBER = 0 # 0 or 1 for WebCam
CONF_THRESHOLD = 0.7
IS_360 = True

cap = cv2.VideoCapture(CAMERA_NUMBER)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

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
        combined_vertical = np.vstack((top_frame, bottom_frame))
        cv2.imshow("Stream", combined_vertical)
    else:
        process_frame(results, frame, conf_thresh=CONF_THRESHOLD)
        cv2.imshow("Stream",frame)

    if cv2.getWindowProperty("Stream", cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or Esc
        break

cap.release()
cv2.destroyAllWindows()