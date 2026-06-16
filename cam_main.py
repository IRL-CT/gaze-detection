#!/usr/bin/env python3
import cv2
import time
import numpy as np
from ultralytics import YOLO
from cam_main_helpers import process_frame
import joblib, pickle

# Load real-time YOLO Pose model (Nano version recommended for speed)
model = YOLO('yolo26n-pose.pt')
classifier = None
scaler = None

CAMERA_NUMBER = 1 # 0 or 1 for WebCam

CONF_THRESHOLD = 0.7
IS_360 = False
IMG_REDUC_FACTOR = 0.6
GAZE_MODEL = True

cap = cv2.VideoCapture(CAMERA_NUMBER)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

if GAZE_MODEL: # See README.md
    try:
        # BEST MODEL: gb_gaze_XXX_2param.pkl
        with open('models/gb_gaze_classifier_2param.pkl', 'rb') as f:
            classifier = pickle.load(f)

        with open('models/gb_gaze_scaler_2param.pkl', 'rb') as f:
            scaler = pickle.load(f)
        # classifier = joblib.load('models/rf_gaze_classifier_v77.joblib')
        # scaler = joblib.load('models/rf_gaze_scaler_v77.joblib')
    except Exception as e:
        print(f"Error loading Gaze Model components: {e}")

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Failed to read camera stream: check CAMERA_NUMBER.")
        time.sleep(0.01)
        continue

    h, w = frame.shape[:2]

    if IMG_REDUC_FACTOR != 1:
        h, w = int(h*IMG_REDUC_FACTOR), int(w*IMG_REDUC_FACTOR)
        frame = cv2.resize(frame, (w, h))
        
    if IS_360:
        top_frame = frame[:h//2, :]
        bottom_frame = frame[h//2:, :]

        top_results = model(top_frame, verbose=False, classes=[0])[0]
        bottom_results= model(bottom_frame, verbose=False, classes=[0])[0]
    else:
        results = model(frame, verbose=False, classes=[0])[0]

    frame_timestamp = time.time()

    if IS_360:
        process_frame(top_results, top_frame, conf_thresh=CONF_THRESHOLD, is_gaze_model=GAZE_MODEL, classifier=classifier, scaler=scaler)
        process_frame(bottom_results, bottom_frame, conf_thresh=CONF_THRESHOLD, is_gaze_model=GAZE_MODEL, classifier=classifier, scaler=scaler)
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