#!/usr/bin/env python3

import cv2
import rclpy
import threading
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# If this helper isn't available, you can comment it out and use fallback drawing
try:
    from cam_main_helpers import process_frame
except ImportError:
    process_frame = None 

CAMERA_NUMBER = 0

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

# ==========================================================
# ROS 2 Node
# ==========================================================
class CameraPublisher(Node):

    def __init__(self):
        super().__init__("camera_publisher")
        
        self.publisher = self.create_publisher(Image, "/camera/image_annotated", 10)
        self.bridge = CvBridge()
        
        # Configuration
        self.IMG_REDUC_FACTOR = 0.4
        self.CONF_THRESHOLD = 0.8
        
        # Initialize camera and model
        self.get_logger().info("Initializing YOLO model and Camera...")
        self.model = YOLO("yolo26n-pose.pt")
        self.cap = cv2.VideoCapture(CAMERA_NUMBER)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera index {CAMERA_NUMBER}")
            return

        # Timer to capture and process frames at ~10 FPS
        self.timer = self.create_timer(1.0 / 10.0, self.process_and_publish) # 10 FPS
        self.get_logger().info("Camera Node initialized successfully.")

    def process_and_publish(self):
        global latest_frame_bytes

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
        bottom_frame = frame[h//2:, :]

        top_results = self.model(top_frame, verbose=False, classes=[0])[0]
        bottom_results = self.model(bottom_frame, verbose=False, classes=[0])[0]

        # Run helper processing if available
        if process_frame:
            process_frame(top_results, top_frame, conf_thresh=self.CONF_THRESHOLD)
            process_frame(bottom_results, bottom_frame, conf_thresh=self.CONF_THRESHOLD)
        else:
            # Fallback visuals if helper isn't present
            cv2.putText(frame, "YOLO Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Re-stitch frames together
        frame = np.vstack((top_frame, bottom_frame))

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
        with frame_lock:
            if latest_frame_bytes is None:
                continue
            frame = latest_frame_bytes
        
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