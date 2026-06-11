#!/usr/bin/env python3
import numpy as np
from math import dist
import cv2

def distance_point_to_line(p1, p2, p3):
    """
    Calculates the distance from point p3 to the line defined by p1 and p2.
    Points should be passed as numpy arrays, e.g., np.array([x, y])
    """
    # Vector of the line segment
    # p1 = np.array(p1)
    # p2 = np.array(p2)
    # p3 = np.array(p3)

    line_vec = p2 - p1
    # Vector from line start to the external point
    point_vec = p3 - p1
    
    # Perpendicular distance formula via 2D cross product & norm
    return abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)

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
            cv2.putText(image, f"Nose Offset: {nose_eyes_offset:.2f}", (25, image.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 0), 4)
            if nose_eyes_offset > 0.15: 
                return False
        # 3. TODO: Check ears, nose, "collinear" (if (distance from earline to the nose/(ear distance) > 0.25?)
            if l_ear[2] > conf_thresh and r_ear[2] > conf_thresh:
                ear_dist = dist(l_ear[:-1], r_ear[:-1])
                ear_nose_dist = distance_point_to_line(l_ear[:-1], r_ear[:-1], nose[:-1])
                ear_nose_offset = ear_nose_dist/ear_dist
                cv2.putText(image, f"Ear Offset: {ear_nose_offset:.2f}", (25, image.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 67, 255), 4)
                if (ear_nose_dist/ear_dist) < 0.25:
                    return True
                
    return False

def process_frame(results, frame, conf_thresh=0.7):
    processed_data = []
    face_names = ["Nose", "L Eye", "R Eye", "L Ear", "R Ear"]
    if results.keypoints is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        # confidences = results.boxes.conf.cpu().numpy()
        kpts_list = results.keypoints.data.cpu().numpy() # Shape: [Num_People, 17, 3]

        for index, kpts in enumerate(kpts_list):
            face_kpts = kpts[0:5] 
            
            approachable = check_approachability(face_kpts, frame, conf_thresh = conf_thresh)
            
            face_serialized = [[float(x), float(y), float(conf)] for x, y, conf in face_kpts]
            bbox = [float(x) for x in boxes[index]]
            
            processed_data.append({
                "person_id": index,
                "bbox": bbox,
                "face_keypoints": face_serialized,
                "approachable": approachable
            })
            
            color = (0, 255, 0) if approachable else (0, 0, 255)
            status = "APPROACHABLE" if approachable else "DNI"
            cv2.putText(frame, status, (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.67, color, 2)
                
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            for idx, pt in enumerate(face_serialized):
                x, y, conf = pt[0], pt[1], pt[2]
                if pt[2] > conf_thresh:
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
