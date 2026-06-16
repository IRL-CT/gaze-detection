#!/usr/bin/env python3

import cv2
import rclpy
import threading
import numpy as np
import time
import pickle
from ultralytics import YOLO
from flask import Flask, Response, render_template_string
from math import dist

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CAMERA_NUMBER = "approachable.mov"
# CAMERA_NUMBER = 0
IS_ORIGINAL_FPS = False
FRAME_RATE = 100

# ==========================================================
# Flask Configuration
# ==========================================================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Gaze Detector Stream</title>
</head>
<body>
    <h1 style="text-align:center;">Gaze Detector Stream</h1>
    <div style="display: flex; justify-content: center; margin: 10px;">
        <img src="/video_feed" width="960">
    </div>
</body>
</html>
"""

# Global variable to safely hold the latest processed frame JPEG bytes across threads
latest_frame_bytes = None
frame_lock = threading.Lock()

def distance_point_to_line(p1, p2, p3):
    """
    Calculates the distance from point p3 to the line defined by p1 and p2.
    Points should be passed as numpy arrays, e.g., np.array([x, y])
    """

    line_vec = p2 - p1
    point_vec = p3 - p1
    
    # Perpendicular distance formula via 2D cross product & norm
    line_vec = np.array([line_vec[0], line_vec[1], 0.0])
    point_vec = np.array([point_vec[0], point_vec[1], 0.0])

    return abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)

def predict_local_image(classifier, scaler, face_kp, conf_thresh=0.7):
    """
    Expects face_kp to be an array/list of shape (5, 3) 
    containing [x, y, confidence] for Nose, L-Eye, R-Eye, L-Ear, R-Ear.
    """
    if classifier is None or scaler is None:
        return False
        
    kp = np.array(face_kp)
    if len(kp) < 5:
        return False

    nose, l_eye, r_eye, l_ear, r_ear = kp[0], kp[1], kp[2], kp[3], kp[4]

    # 0. Is Nose visible?
    if nose[2] < conf_thresh:
        return False

    nose_eyes_offset = -1.0
    ear_nose_offset = -1.0

    # 1. Calculate Nose-to-Eyes Horizontal Offset
    if l_eye[2] > conf_thresh and r_eye[2] > conf_thresh:
        eye_center_x = (l_eye[0] + r_eye[0]) / 2.0
        eye_distance = np.abs(l_eye[0] - r_eye[0])
        
        if eye_distance > 0:
            nose_eyes_offset = np.abs(nose[0] - eye_center_x) / eye_distance

    # 2. Calculate Ear-to-Nose Perpendicular Offset
    if l_ear[2] > conf_thresh and r_ear[2] > conf_thresh:
        ear_dist = dist(l_ear[:-1], r_ear[:-1])
        ear_nose_dist = distance_point_to_line(l_ear[:-1], r_ear[:-1], nose[:-1])

        if ear_dist > 0:
            ear_nose_offset = ear_nose_dist / ear_dist

    # 3. Calculate Nose-Eyeline Ratio
    l_eye_nose_dist = (l_eye[0] - nose[0])
    r_eye_nose_dist = (r_eye[0] - nose[0])
    nose_eyeline_ratio = abs(l_eye_nose_dist/r_eye_nose_dist)

    # 4. Number of Ears visible
    num_ears = 0
    if r_ear[2] > 0.8:
        num_ears+=1
    if l_ear[2] > 0.8:
        num_ears+=1

    # 5. Scale & Predict

    # feature_vector = np.array([nose_eyes_offset, ear_nose_offset, nose_eyeline_ratio]).reshape(1,-1)
    feature_vector = np.array([nose_eyes_offset, ear_nose_offset]).reshape(1, -1)

    scaled_kp = scaler.transform(feature_vector)
    prediction = classifier.predict(scaled_kp)[0]
    
    return (prediction == 1)

def check_approachability(keypoints, image, conf_thresh=0.7, reduc=1):
    """
    Baseline Behavior Layer: Evaluates if a person is looking towards the camera 
    from a high angle / top-down perspective.
    """

    nose = keypoints[0]
    l_eye = keypoints[1]
    r_eye = keypoints[2]
    l_ear = keypoints[3]
    r_ear = keypoints[4]
    
    # 1. Is nose visible?
    if nose[2] < conf_thresh:
        return False
        
    # 2. Is nose relatively centered betwixt eyes?
    if l_eye[2] > conf_thresh and r_eye[2] > conf_thresh:
        eye_center_x = (l_eye[0] + r_eye[0]) / 2
        eye_distance = abs(l_eye[0] - r_eye[0])
        
        if eye_distance > 0:
            nose_eyes_offset = abs(nose[0] - eye_center_x) / eye_distance
            cv2.putText(image, f"Nose Offset: {nose_eyes_offset:.2f}", (25, int(image.shape[0] - 50*reduc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.67*reduc, (255, 255, 0), 2)
            if nose_eyes_offset > 0.15: 
                return False
        # 3. Check ears, nose, "collinear" (if (distance from earline to the nose/(ear distance) > 0.25?)
            if l_ear[2] > conf_thresh and r_ear[2] > conf_thresh:
                ear_dist = dist(l_ear[:-1], r_ear[:-1])
                ear_nose_dist = distance_point_to_line(l_ear[:-1], r_ear[:-1], nose[:-1])
                ear_nose_offset = ear_nose_dist/ear_dist
                cv2.putText(image, f"Ear Offset: {ear_nose_offset:.2f}", (25, int(image.shape[0] - 30*reduc)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.67*reduc, (0, 67, 255), 2)
                if (ear_nose_dist/ear_dist) < 0.25:
                    return True
                
    return False

def process_frame(classifier, scaler, results, frame, conf_thresh=0.7, reduc=1):
    processed_data = []
    # Filter out low confidence detections
    if results.keypoints is not None and len(results.keypoints.xy) > 0:
        boxes = []
        for box in results.boxes:
            if box.conf.item() > conf_thresh:
                boxes.append(box.xyxy.cpu().numpy()[0])
        kpts_list = results.keypoints.data.cpu().numpy()

        for index, kpts in enumerate(kpts_list):
            if index >= len(boxes):
                break
            face_kpts = kpts[0:5]
            
            approachable = predict_local_image(classifier, scaler, face_kpts)
            
            face_serialized = [[float(x), float(y), float(conf)] for x, y, conf in face_kpts]
            bbox = [float(x) for x in boxes[index]]
            
            # TODO: Check if helpful data
            processed_data.append({
                "person_id": index,
                "bbox": bbox,
                "face_keypoints": face_serialized,
                "approachable": approachable
            })
            
            color = (0, 255, 0) if approachable else (0, 0, 255)
            status = "APPROACH" if approachable else "DNI"

            cv2.putText(frame, status, (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.67*reduc, color, 2)
                
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

            for idx, pt in enumerate(face_serialized):
                if pt[2] > conf_thresh:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)
                        # cv2.putText( frame, f"({pt[0]:.2f}, {pt[1]:.2f})", (int(pt[0])+5, int(pt[1])-5), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
                        # cv2.putText( frame, f"{conf:.2f}", (int(pt[0])+5, int(pt[1])-20),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.67, (67,255,67), 1)

# ==========================================================
# ROS 2 Node
# ==========================================================
class CameraPublisher(Node):

    def __init__(self):
        
        self.prev = time.time()
        super().__init__("camera_publisher")
        
        self.publisher = self.create_publisher(Image, "/camera/image_annotated", 10)
        self.bridge = CvBridge()
        
        # Configuration
        self.IMG_REDUC_FACTOR = 0.5
        self.CONF_THRESHOLD = 0.8
        
        # Initialize camera and models
        self.get_logger().info("Initializing YOLO model and Camera...")
        self.model = YOLO("yolo26n-pose.pt")

        self.classifier = None
        self.scaler = None

        # TODO: generalize filepaths?
        with open('models/gb_gaze_classifier_2param.pkl', 'rb') as f:
            self.classifier = pickle.load(f)

        with open('models/gb_gaze_scaler_2param.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        # classifier = joblib.load('models/rf_gaze_classifier_v77.joblib')
        # scaler = joblib.load('models/rf_gaze_scaler_v77.joblib')

        self.cap = cv2.VideoCapture(CAMERA_NUMBER)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera index {CAMERA_NUMBER}")
            return

        # Timer to capture and process frames at ~10 FPS
        self.timer = self.create_timer(1.0 / 10, self.process_and_publish) # 10 FPS
        self.get_logger().info("Camera Node initialized successfully.")

    def process_and_publish(self):
        global latest_frame_bytes
        global time_elapsed 

        time_elapsed = time.time() - self.prev
        if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / FRAME_RATE):
            self.prev = time.time()
            success, frame = self.cap.read()
            if not success:
                self.get_logger().warning("Failed to read camera frame")
                return

            h, w = frame.shape[:2]

            # Resize if factor is modified
            if self.IMG_REDUC_FACTOR != 1:
                h, w = int(h * self.IMG_REDUC_FACTOR), int(w * self.IMG_REDUC_FACTOR)
                frame = cv2.resize(frame, (w, h))

            # Split frame for independent top/bottom inference
            top_frame = frame[:h//2, :]
            # bottom_frame = frame[h//2:, :]

            start = time.time()
            #TODO: see whether limiting subjects is necessary (max_det = N)
            top_results = self.model(top_frame, verbose=False, classes=[0])[0]
            print("YOLO:", time.time() - start)

            # top_results = self.model(top_frame, verbose=False, classes=[0])[0]
            # bottom_results = self.model(bottom_frame, verbose=False, classes=[0])[0]

            # Run helper processing if available
            if process_frame:
                process_frame(self.classifier, self.scaler, top_results, top_frame, conf_thresh=self.CONF_THRESHOLD, reduc=self.IMG_REDUC_FACTOR)
                # process_frame(self.classifier, self.scaler, bottom_results, bottom_frame, conf_thresh=self.CONF_THRESHOLD, reduc=self.IMG_REDUC_FACTOR)
            else:
                # Fallback visuals if helper isn't present
                cv2.putText(frame, "YOLO Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Re-stitch frames together
            frame = top_frame
            # frame = np.vstack((top_frame, bottom_frame))

            # 1. Update the Flask stream buffer (Thread Safe)
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                with frame_lock:
                    latest_frame_bytes = buffer.tobytes()

            # 2. Publish to ROS 2 Topic
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "camera"
                self.publisher.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish image: {str(e)}")
        else:
            # Prevent hot-spin while waiting for next frame/time budget.
            # Without this, the loop burns CPU doing no useful work.
            time.sleep(0.001)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

# ==========================================================
# Flask Routes & Generators
# ==========================================================
@app.route("/")
def index():
    return render_template_string(HTML)

def generate_frames():
    """Generator function that pulls the latest frame bytes for Flask."""
    global latest_frame_bytes
    while True:
        if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / FRAME_RATE):
            with frame_lock:
                if latest_frame_bytes is None:
                    continue
                frame = latest_frame_bytes
            
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        else:
            # Prevent hot-spin while waiting for next frame/time budget.
            # Without this, the loop burns CPU doing no useful work.
            time.sleep(0.001)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ==========================================================
# Main Execution Entrypoint
# ==========================================================
def main():
    rclpy.init()
    node = CameraPublisher()

    # Spin ROS 2 in a background thread so it doesn't block Flask
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        # Run Flask on the main thread
        # threaded=True allows handling multiple web clients seamlessly
        app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.get_logger().info("Shutting down cleanly...")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()