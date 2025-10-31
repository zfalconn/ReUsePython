import math
import numpy as np

CONTROL_HZ = 20.0 
DEADBAND_M = 0.005   
MAX_STEP_M = 0.005  

def tf_camera_to_gripper(point_cam, 
                        R_gc = np.array([
                            [0, -1, 0],
                            [0,  0, 1],
                            [-1, 0, 0]
                        ]), 
                        t_gc = np.array([0.085, -0.220, 0.040])):
    """
    Transform a 3D point from the camera frame to the gripper frame.

    Args:
        point_cam (array-like): [x, y, z] in camera frame
        R_gc (np.ndarray): 3x3 rotation matrix from camera to gripper
        t_gc (array-like): 3x1 translation vector from camera to gripper

    Returns:
        np.ndarray: [x, y, z] in gripper frame
    """
    point_cam = np.array(point_cam).reshape(3, 1)
    R_gc = np.array(R_gc).reshape(3, 3)
    t_gc = np.array(t_gc).reshape(3, 1)

    point_gripper = R_gc @ point_cam + t_gc
    return point_gripper.flatten()

def select_target(detections, target_class):
    filtered = [d for d in detections if d["class_name"] == target_class]
    if not filtered:
        return None
    # pick closest to image center
    target_detection = min(filtered, key=lambda d: (d["center_2d"][0]-640)**2 + (d["center_2d"][1]-360)**2)
    return target_detection


def rotate_point_to_gripper(target_detection):
    target_detection["xyz_gripper_frame"] = tf_camera_to_gripper(target_detection["xyz"],
    t_gc=np.array([0.0, 0.0, 0.0])) #only rotate frame, no translation
    return target_detection

def apply_deadband_and_limits(prev, desired):
    delta = desired - prev
    # deadband
    if np.linalg.norm(delta) < DEADBAND_M:
        return prev, np.zeros(3)
    # limit step magnitude
    norm = np.linalg.norm(delta)
    if norm > MAX_STEP_M:
        delta = delta / norm * MAX_STEP_M
    new = prev + delta
    return new, delta