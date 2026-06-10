import cv2
import time
import numpy as np
# from flask import Flask, render_template
# from flask_socketio import SocketIO
from ultralytics import YOLO
from cam_main_helpers import check_approachability, distance_point_to_line

# Load real-time YOLO Pose model (Nano version recommended for speed)
model = YOLO('yolo26n-pose.pt')

# TODO: connect & test w/Insta360, 360 view
CAMERA_NUMBER = 1 # 0 or 1 for WebCam
CONF_THRESHOLD = 0.7

cap = cv2.VideoCapture(CAMERA_NUMBER)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

face_names = ["Nose", "L Eye", "R Eye", "L Ear", "R Ear"]

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        print("Failed to read camera stream: check CAMERA_NUMBER.")
        time.sleep(0.01)
        continue

    frame_timestamp = time.time()

    # Run inference 
    results = model(frame, verbose=False, classes=[0])[0]

    processed_data = []
    
    if results.keypoints is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        kpts_list = results.keypoints.data.cpu().numpy() # Shape: [Num_People, 17, 3]

        for index, kpts in enumerate(kpts_list):
            face_kpts = kpts[0:5] 
            
            approachable = check_approachability(face_kpts, frame, conf_thresh = CONF_THRESHOLD)
            
            face_serialized = [[float(x), float(y), float(conf)] for x, y, conf in face_kpts]
            bbox = [float(x) for x in boxes[index]]
            
            processed_data.append({
                "person_id": index,
                "bbox": bbox,
                "face_keypoints": face_serialized,
                "approachable": approachable
            })
            
            color = (0, 255, 0) if approachable else (0, 0, 255)
            if approachable:
                cv2.putText(frame, "APPROACHABLE", (25, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 255, 0), 2)
                
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            for idx, pt in enumerate(face_serialized):
                x, y, conf = pt[0], pt[1], pt[2]
                if pt[2] > CONF_THRESHOLD:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)
                    cv2.putText(
                    frame,
                    f"{face_names[idx]} ({x:.2f}, {y:.2f})",
                    (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255,255,0),
                    1)
                    cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (int(x)+5, int(y)-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (67,255,67),
                    1)
                # TODO: integrate this better here 
                
        cv2.imshow("Stream",frame)

        if cv2.getWindowProperty("Stream", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or Esc
            break

cap.release()
cv2.destroyAllWindows()