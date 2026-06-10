import numpy as np
import cv2

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
            nose_offset = abs(nose[0] - eye_center_x) / eye_distance
            cv2.putText(image, f"Nose Offset: {nose_offset:.2f}", (25, image.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 0), 4)
            if nose_offset < 0.15: 
                return True
                
    # 3. TODO: Check ears, nose, "collinear" (if (distance from earline to the nose/(ear distance) > 0.5?)

    return False

def distance_point_to_line(p1, p2, p3):
    """
    Calculates the distance from point p3 to the line defined by p1 and p2.
    Points should be passed as numpy arrays, e.g., np.array([x, y])
    """
    # Vector of the line segment
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    line_vec = p2 - p1
    # Vector from line start to the external point
    point_vec = p3 - p1
    
    # Perpendicular distance formula via 2D cross product & norm
    return abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)
