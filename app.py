import cv2
import time
import numpy as np
import joblib
from flask import Flask, Response, render_template_string
from math import dist
from ultralytics import YOLO

# ========================================================== #
# Configuration
# ========================================================== #
CAMERA_NUMBER = "test_videos/test.mov"

IS_360 = False
GAZE_MODEL = True
IMG_REDUC_FACTOR = 0.6
MORE_ANNOTATIONS = False

# Load models safely
model = YOLO("yolov8n-pose.pt") 

classifier = None
scaler = None
if GAZE_MODEL:
    try:
        classifier = joblib.load('gaze_classifier_2.joblib')
        scaler = joblib.load('gaze_scaler_2.joblib')
    except Exception as e:
        print(f"Error loading Gaze Model components: {e}")

# Open the video capture once at global initialization
cap = cv2.VideoCapture(CAMERA_NUMBER)
if not cap.isOpened():
    print(f"\n[CRITICAL ERROR] Cannot open video source: '{CAMERA_NUMBER}'")
    print("Ensure the file exists, the path is correct, or try using a webcam integer like 0.\n")

face_names = ["Nose", "L Eye", "R Eye", "L Ear", "R Ear"]

# ========================================================== #
# Flask App
# ========================================================== #
app = Flask(__name__)
HTML = """
<!DOCTYPE html>
<html>
<head><title>Gaze Detector Stream</title></head>
<body style="background:#111; color:white; font-family:sans-serif; text-align:center;">
    <h1>Gaze Detector Stream</h1>
    <div style="display: flex; justify-content: center; margin: 10px;">
        <img src="/video_feed" width="960" style="border: 2px solid #333;">
    </div>
</body>
</html>
"""

def predict_local_image(face_kp):
    if classifier is None or scaler is None:
        return False
    face_flat = np.array(face_kp).flatten().reshape(1, -1)
    scaled_kp = scaler.transform(face_flat)
    prediction = classifier.predict(scaled_kp)[0]
    return (prediction == 1)

def distance_point_to_line(p1, p2, p3):
    line_vec = p2 - p1
    point_vec = p3 - p1
    line_vec_3d = np.array([line_vec[0], line_vec[1], 0.0])
    point_vec_3d = np.array([point_vec[0], point_vec[1], 0.0])
    norm = np.linalg.norm(line_vec_3d)
    if norm == 0:
        return 0
    return abs(np.cross(line_vec_3d, point_vec_3d)[2]) / norm

def check_approachability(keypoints, image, conf_thresh=0.7):
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
    # [x, y, confidence]
    if nose[2] < conf_thresh:
        return False
        
    # 2. Is nose relatively centered betwixt eyes?
    if l_eye[2] > conf_thresh and r_eye[2] > conf_thresh:
        eye_center_x = (l_eye[0] + r_eye[0]) / 2
        eye_distance = abs(l_eye[0] - r_eye[0])
        
        if eye_distance > 0:
            nose_eyes_offset = abs(nose[0] - eye_center_x) / eye_distance
            if nose_eyes_offset > 0.15: 
                return False
            if MORE_ANNOTATIONS:
                cv2.putText(image, f"Nose Offset: {nose_eyes_offset:.2f}", (25, image.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 0), 4)
            
        # 3. Check ears, nose, "collinear" (if (distance from earline to the nose/(ear distance) > 0.25?)
            if l_ear[2] > conf_thresh and r_ear[2] > conf_thresh:
                ear_dist = dist(l_ear[:-1], r_ear[:-1])
                ear_nose_dist = distance_point_to_line(l_ear[:-1], r_ear[:-1], nose[:-1])
                ear_nose_offset = ear_nose_dist/ear_dist
                if ear_nose_offset < 0.25:
                    return True
                if MORE_ANNOTATIONS:
                    cv2.putText(image, f"Ear Offset: {ear_nose_offset:.2f}", (25, image.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 67, 255), 4)
                
            else:
                return True
                
    return False

def process_frame(results, frame, conf_thresh=0.7):
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
            
            if GAZE_MODEL:
                face_kp = [sublist[:-1] for sublist in face_kpts]
                approachable = predict_local_image(face_kp)
            else:
                approachable = check_approachability(face_kpts, frame, conf_thresh=conf_thresh)
            
            face_serialized = [[float(x), float(y), float(conf)] for x, y, conf in face_kpts]
            bbox = [float(x) for x in boxes[index]]
            
            color = (0, 255, 0) if approachable else (0, 0, 255)
            status = "APPROACH" if approachable else "DNI"
            
            cv2.putText(frame, status, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

            # for box in results.boxes:
            #     conf_value = box.conf.item() 
            #     cv2.putText( frame, f"{conf_value:.2f}", (int(bbox[0])+10, int(bbox[1])-10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255,0,255), 1)
            
            if MORE_ANNOTATIONS:
                for pt in face_serialized:
                    x, y, conf = pt[0], pt[1], pt[2]
                    if conf > conf_thresh:
                        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 0), -1)
                        # cv2.putText( frame, f"({x:.2f}, {y:.2f})", (int(x)+5, int(y)-5), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
                        # cv2.putText( frame, f"{conf:.2f}", (int(x)+5, int(y)-20),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.67, (67,255,67), 1)

# ========================================================== #
# Frame Generator
# ========================================================== #
def generate_frames():
    while True:
        if not cap.isOpened():
            # If capture device completely failed initialization, stream a black placeholder image
            print("[WARN] Video capture is not open. Streaming empty frame placeholder.")
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "VIDEO PATH ERROR / DISCONNECTED", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode(".jpg", black_frame)
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(1)
            continue

        success, frame = cap.read()
        if not success:
            print("Video stream finished or failed. Restarting video track...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.5)
            continue

        h, w = frame.shape[:2]
        if IMG_REDUC_FACTOR != 1:
            h, w = int(h * IMG_REDUC_FACTOR), int(w * IMG_REDUC_FACTOR)
            frame = cv2.resize(frame, (w, h))

        if IS_360:
            # Enforce an even slice layout split to preserve shape constraints across channels
            half_h = h // 2
            top_frame = frame[:half_h, :]
            bottom_frame = frame[half_h:(half_h * 2), :] # Guarantees exact structural matching dimensions
            
            top_results = model(top_frame, verbose=False, classes=[0])[0]
            bottom_results = model(bottom_frame, verbose=False, classes=[0])[0]
            
            process_frame(top_results, top_frame)
            process_frame(bottom_results, bottom_frame)
            frame = np.vstack((top_frame, bottom_frame))
        else:
            results = model(frame, verbose=False, classes=[0])[0]
            process_frame(results, frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
            
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# ========================================================== #
# Routes
# ========================================================== #
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)
