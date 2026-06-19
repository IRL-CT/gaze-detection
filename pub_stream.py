#!/usr/bin/env python3

import cv2
import rclpy
import signal
import threading
import numpy as np
import time
import json
import pickle
import os
import torch
# import joblib
from ultralytics import YOLO
from flask import Flask, Response, render_template_string
from math import dist

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# Resolve paths relative to this file so the node runs from any working dir.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# [0..2] for a live camera, or a path to a video file.
CAMERA_NUMBER = os.path.join(SCRIPT_DIR, "test_videos", "test.mov")
# CAMERA_NUMBER = 2
# CAMERA_NUMBER = "test_videos/test.mov"
# CAMERA_NUMBER = 0
DUAL = False                    # True to isolate top view (faster processing)

# YOLO model + inference device
MODEL_PATH = os.path.join(SCRIPT_DIR, "yolo11n-pose.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()   # FP16 inference is a free speedup on Jetson GPU

IS_ORIGINAL_FPS = False
FRAME_RATE = 30 if not IS_ORIGINAL_FPS else 100
MORE_ANNOTATIONS = True
SMOOTHING = True

IMG_REDUC_FACTOR = 0.5
CONF_THRESHOLD = 0.7

# Set before the first frame is processed so the Flask generator never hits a NameError.
time_elapsed = 0.0

# SMOOTHING
# Tuning Parameters
ALPHA = 0.25          # EMA weight on the current frame (lower = smoother, more lag)
DISTANCE_THRESH = 80  # Max px a person's box center can move between frames to match

# Hysteresis: require strong, sustained evidence to flip state. A single 0.5
# threshold chatters; these two create a "dead zone" the smoothed value must fully
# cross before the label changes.
ENTER_THRESH = 0.60   # smoothed prob must rise above this to become APPROACH
EXIT_THRESH = 0.40    # ...and fall below this to drop back to DNI
TRACK_TTL = 8         # frames to coast a track after it stops being detected

# Structure: { person_id: {'bbox': [...], 'smoothed_state': float,
#                          'approachable': bool, 'missed': int} }
active_tracks = {}
next_track_id = 0

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
    Calculate distance from line created by [p1, p2] to p3.
    """
    line_vec = p2 - p1
    point_vec = p3 - p1
    line_vec_3d = np.array([line_vec[0], line_vec[1], 0.0])
    point_vec_3d = np.array([point_vec[0], point_vec[1], 0.0])
    norm = np.linalg.norm(line_vec_3d)
    return 0 if norm == 0 else abs(np.cross(line_vec_3d, point_vec_3d)[2]) / norm

def predict_local_image(classifier, scaler, face_kp):
    """
    Expects face_kp to be an array/list of shape (5, 3)
    containing [x, y, confidence] for Nose, L-Eye, R-Eye, L-Ear, R-Ear.

    Returns the probability in [0, 1] that the person is looking (approachable),
    or None when there is no usable measurement (no model / malformed keypoints)
    so the caller can hold the prior smoothed value instead of treating a missing
    reading as "not looking".
    """
    if classifier is None or scaler is None:
        return None

    kp = np.array(face_kp)
    if len(kp) < 5:
        return None

    nose, l_eye, r_eye, l_ear, r_ear = kp[0], kp[1], kp[2], kp[3], kp[4]

    # 0. Is Nose visible? An absent frontal nose is evidence they are turned away,
    #    so report 0.0 (not looking) rather than "no data".
    if nose[2] < CONF_THRESHOLD:
        return 0.0

    nose_eyes_offset = -1.0
    ear_nose_offset = -1.0

    # 1. Calculate Nose-to-Eyes Horizontal Offset
    if l_eye[2] > CONF_THRESHOLD and r_eye[2] > CONF_THRESHOLD:
        eye_center_x = (l_eye[0] + r_eye[0]) / 2.0
        eye_distance = np.abs(l_eye[0] - r_eye[0])
        
        if eye_distance > 0:
            nose_eyes_offset = np.abs(nose[0] - eye_center_x) / eye_distance

    # 2. Calculate Ear-to-Nose Perpendicular Offset
    if l_ear[2] > CONF_THRESHOLD and r_ear[2] > CONF_THRESHOLD:
        ear_dist = dist(l_ear[:-1], r_ear[:-1])
        ear_nose_dist = distance_point_to_line(l_ear[:-1], r_ear[:-1], nose[:-1])

        if ear_dist > 0:
            ear_nose_offset = ear_nose_dist / ear_dist

    # 3. Calculate Nose-Eyeline Ratio
    l_eye_nose_dist = (l_eye[0] - nose[0])
    r_eye_nose_dist = (r_eye[0] - nose[0])
    # Guard against a nose horizontally aligned with the right eye (0 denominator),
    # which would yield inf/NaN and crash the scaler downstream.
    if r_eye_nose_dist == 0:
        return 0.0
    nose_eyeline_ratio = abs(l_eye_nose_dist/r_eye_nose_dist)

    # 4. Number of Ears visible
    num_ears = 0
    if r_ear[2] > 0.8:
        num_ears+=1
    if l_ear[2] > 0.8:
        num_ears+=1

    # 5. Scale & Predict
    feature_vector = np.array([nose_eyes_offset, ear_nose_offset, nose_eyeline_ratio]).reshape(1,-1)
    # feature_vector = np.array([nose_eyes_offset, ear_nose_offset]).reshape(1, -1)

    scaled_kp = scaler.transform(feature_vector)

    # Return a continuous probability so the EMA smooths a soft signal instead of a
    # noisy hard 0/1 (the main source of green/red flicker). Fall back to the binary
    # decision if this classifier has no predict_proba.
    if hasattr(classifier, "predict_proba"):
        return float(classifier.predict_proba(scaled_kp)[0][1])
    return float(classifier.predict(scaled_kp)[0] == 1)

def smooth_process_frame(classifier, scaler, results, frame):
    global active_tracks, next_track_id

    boxes = []
    kpts_list = []
    if results.keypoints is not None and len(results.keypoints.xy) > 0:
        for box in results.boxes:
            if box.conf.item() > CONF_THRESHOLD:
                boxes.append(box.xyxy.cpu().numpy()[0])
        kpts_list = results.keypoints.data.cpu().numpy()

    new_tracks = {}
    matched_ids = set()
    detections = []  # Per-person results returned to the caller for ROS publishing
    
    top_index = 0
    top_smoothed_val = 0

    for index, kpts in enumerate(kpts_list):
        if index >= len(boxes):
            break
        face_kpts = kpts[0:5]
        bbox = [float(x) for x in boxes[index]]
        smoothed_val = None

        if not SMOOTHING:
            prob = predict_local_image(classifier, scaler, face_kpts)
            approachable = prob is not None and prob > 0.5
        else:
            current_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

            # Raw per-frame probability (None == no usable face this frame)
            raw_prob = predict_local_image(classifier, scaler, face_kpts)

            # Spatial matching: nearest still-unclaimed track from prior frames
            matched_id = None
            min_dist = float('inf')
            for tid, tdata in active_tracks.items():
                if tid in matched_ids:
                    continue
                last_bbox = tdata['bbox']
                last_center = np.array([(last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2])
                d = np.linalg.norm(current_center - last_center)
                if d < min_dist and d < DISTANCE_THRESH:
                    min_dist = d
                    matched_id = tid

            if matched_id is None:
                # New person entering the scene; seed the EMA with this reading
                # (or a neutral 0.5 if we have no reading yet).
                matched_id = next_track_id
                next_track_id += 1
                smoothed_val = raw_prob if raw_prob is not None else 0.5
                approachable = smoothed_val > ENTER_THRESH
            else:
                matched_ids.add(matched_id)
                prior = active_tracks[matched_id]
                if raw_prob is None:
                    # No usable reading: coast on the prior value so a momentary
                    # keypoint dropout doesn't yank the state around.
                    smoothed_val = prior['smoothed_state']
                else:
                    # EMA: smoothed = alpha*current + (1-alpha)*history
                    smoothed_val = (ALPHA * raw_prob) + ((1.0 - ALPHA) * prior['smoothed_state'])

                # Hysteresis: only flip the label when the smoothed value fully
                # crosses the far threshold, so borderline values can't chatter.
                if prior['approachable']:
                    approachable = smoothed_val > EXIT_THRESH
                else:
                    approachable = smoothed_val > ENTER_THRESH

            new_tracks[matched_id] = {
                'bbox': bbox,
                'smoothed_state': smoothed_val,
                'approachable': approachable,
                'missed': 0,
            }

            top_smoothed_val = smoothed_val if smoothed_val > top_smoothed_val and approachable else top_smoothed_val

        # Draw overlays
        color = (0, 255, 0) if approachable else (0, 0, 255)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        

        if MORE_ANNOTATIONS and SMOOTHING:
            status = f"APPROACH ({smoothed_val:.2f})" if approachable else f"DNI ({smoothed_val:.2f})"
            cv2.putText(frame, status, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.85*IMG_REDUC_FACTOR, color, 1)

        # Collect this person's result for publishing. id is the stable track id
        # when smoothing, else the per-frame detection index.
        detections.append({
            "id": int(matched_id) if SMOOTHING else int(index),
            "bbox": [round(float(v), 1) for v in bbox],
            "approachable": bool(approachable),
            "smoothed_state": round(float(smoothed_val), 3) if smoothed_val is not None else None,
        })

    # Coast tracks that weren't matched this frame so brief detection gaps don't
    # reset their smoothed state; drop them once they've been missing too long.
    if SMOOTHING:
        if top_index < len(boxes):
            bbox = [float(x) for x in boxes[top_index]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)

        for tid, tdata in active_tracks.items():
            if tid in new_tracks:
                continue
            tdata['missed'] = tdata.get('missed', 0) + 1
            if tdata['missed'] <= TRACK_TTL:
                new_tracks[tid] = tdata
        active_tracks = new_tracks

    return detections

# ==========================================================
# ROS 2 Node
# ==========================================================
class CameraPublisher(Node):

    def __init__(self):
        
        self.prev = time.time()
        super().__init__("camera_publisher")
        
        self.publisher = self.create_publisher(Image, "/camera/image_annotated", 10)
        # Per-person gaze results as a JSON string (no custom msg package needed).
        self.gaze_publisher = self.create_publisher(String, "/camera/gaze_data", 10)
        self.bridge = CvBridge()
        
        # Initialize camera and models
        self.get_logger().info(f"Initializing YOLO model on '{DEVICE}' and Camera...")
        self.model = YOLO(MODEL_PATH)
        self.model.to(DEVICE)
        # Warm up the GPU graph so the first real frame isn't slow.
        self.model(np.zeros((64, 64, 3), dtype=np.uint8),
                   device=DEVICE, half=USE_HALF, verbose=False)
        self.get_logger().info(f"YOLO model loaded: {os.path.basename(MODEL_PATH)} on {DEVICE}")

        self.classifier = None
        self.scaler = None

        clf_path = os.path.join(SCRIPT_DIR, 'models', 'gb_gaze_classifier_3param_ros.pkl')
        scl_path = os.path.join(SCRIPT_DIR, 'models', 'gb_gaze_scaler_3param_ros.pkl')
        with open(clf_path, 'rb') as f:
            self.classifier = pickle.load(f)

        with open(scl_path, 'rb') as f:
            self.scaler = pickle.load(f)
        # classifier = joblib.load('models/gaze_classifier.joblib')
        # scaler = joblib.load('models/gaze_classifier.joblib')

        self.cap = cv2.VideoCapture(CAMERA_NUMBER)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera index {CAMERA_NUMBER}")
            return

        # Drive capture/inference at the target frame rate (GPU keeps up; the
        # FRAME_RATE gate inside the callback throttles actual work).
        self.timer = self.create_timer(1.0 / FRAME_RATE, self.process_and_publish)
        self.get_logger().info("Camera Node initialized successfully.")

    def process_and_publish(self):
        global latest_frame_bytes
        global time_elapsed

        # Bail out once the context is shutting down so we don't publish into an
        # invalid context (the source of the "Failed to publish image" errors on Ctrl+C).
        if not rclpy.ok():
            return

        time_elapsed = time.time() - self.prev
        if (IS_ORIGINAL_FPS or time_elapsed > 1.0 / FRAME_RATE):
            self.prev = time.time()
            success, frame = self.cap.read()
            if not success:
                # Loop video files; for a live camera this is a transient read miss.
                if isinstance(CAMERA_NUMBER, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.get_logger().warning("Failed to read camera frame")
                return

            h, w = frame.shape[:2]

            # Resize if factor is modified
            if IMG_REDUC_FACTOR != 1:
                h, w = int(h * IMG_REDUC_FACTOR), int(w * IMG_REDUC_FACTOR)
                frame = cv2.resize(frame, (w, h))

            # Split frame for independent top/bottom inference
            top_frame = frame[:h//2, :] if DUAL else frame
            top_results = self.model(top_frame, verbose=False, classes=[0],
                                     device=DEVICE, half=USE_HALF)[0]
            if DUAL:
                bottom_frame = frame[h//2:, :]
                bottom_results = self.model(bottom_frame, verbose=False, classes=[0],
                                            device=DEVICE, half=USE_HALF)[0]

            # Run helper processing if available
            detections = []
            if smooth_process_frame:
                detections = smooth_process_frame(self.classifier, self.scaler, top_results, top_frame)
                if DUAL:
                    bottom_dets = smooth_process_frame(self.classifier, self.scaler, bottom_results, bottom_frame)
                    # Shift bottom-half boxes back into full-frame coordinates.
                    for det in bottom_dets:
                        det["bbox"][1] += h // 2
                        det["bbox"][3] += h // 2
                    detections.extend(bottom_dets)
            else:
                # Fallback visuals if helper isn't present
                cv2.putText(frame, "YOLO Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Re-stitch frames together
            frame = np.vstack((top_frame, bottom_frame)) if DUAL else top_frame

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
                self.publisher.publish(msg) # Publish image to ROS topic /camera/image_annotated

                # 3. Publish per-person gaze results to /camera/gaze_data as JSON.
                gaze_msg = String()
                gaze_msg.data = json.dumps({
                    "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    "smoothing": SMOOTHING,
                    "people": detections,
                })
                self.gaze_publisher.publish(gaze_msg)
            except Exception as e:
                # During Ctrl+C the context can be torn down between the rclpy.ok()
                # check above and this publish, raising "publisher's context is
                # invalid". That's a benign shutdown race, not a real failure, so
                # only surface the error while the context is still valid.
                if rclpy.ok():
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
    """Generator function that pulls the latest frame bytes for Flask.

    Only emits when a *new* frame is available (each processed frame is a fresh
    bytes object), so we don't flood the client with duplicate JPEGs.
    """
    global latest_frame_bytes
    last_sent = None
    while True:
        with frame_lock:
            frame = latest_frame_bytes
        if frame is None or frame is last_sent:
            # No new frame yet; yield CPU/bandwidth instead of hot-spinning.
            time.sleep(0.005)
            continue
        last_sent = frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

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

    def shutdown(signum=None, frame=None):
        # Fast, clean shutdown. rclpy.init() installs its own SIGINT handler that
        # can swallow Ctrl+C before werkzeug ever sees it, leaving the Flask server
        # (and port 5050) alive while the ROS context tears down underneath it.
        # We override that handler here so Ctrl+C reliably reaches this teardown.
        # The /video_feed route is an endless generator and the ROS executor runs
        # in a daemon thread, so a "polite" teardown can hang for seconds waiting
        # on them. Stop ROS, release the camera (the one resource the OS won't
        # reclaim cleanly), then hard-exit and let the OS reap the rest (sockets,
        # daemon threads, GPU context) instantly.
        node.get_logger().info("Shutting down...")
        if rclpy.ok():
            rclpy.shutdown()  # stops the spin thread so its callbacks stop firing
        cap = getattr(node, "cap", None)
        if cap is not None and cap.isOpened():
            cap.release()
        os._exit(0)

    # Override rclpy's handlers so Ctrl+C (and SIGTERM) reach our teardown.
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Run Flask on the main thread.
    # threaded=True allows handling multiple web clients seamlessly
    app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)

if __name__ == "__main__":
    main()