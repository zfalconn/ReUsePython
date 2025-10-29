from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
from collections import Counter
# import sys
# import os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def detection_xyz(model: YOLO, color_frame, depth_frame, intrinsics, img_width, img_height, **yolo_args):

    color_image = np.asanyarray(color_frame.get_data())

    results = model(color_image, **yolo_args)[0]
    detections = []
    if results.boxes:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cx = max(0, min(cx, img_width - 1))
            cy = max(0, min(cy, img_height - 1))

            # Depth
            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue

            # 3D point
            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
            print(f"POINT 3D:  {point_3d} --- {cls}")

            point_3d_gripper = tf_camera_to_gripper(point_3d, 
                                                    t_gc=np.array([0.0, 0.0, 0.0]))
            print(f"POINT 3D in gripper's frame: {point_3d_gripper} --- {cls}")
            detections.append({
                "class_id": cls,
                "class_name": model.names[cls],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center_2d": [cx, cy],
                "xyz": point_3d ,
                "xyz_gripper_frame": point_3d_gripper
            })

    return detections

def postprocess_detection(detections, depth_frame, intrinsics):
    class_counts = Counter(det["class_name"] for det in detections)
    terminal_detections = [det for det in detections if det["class_name"] == "terminal"]
    if class_counts["terminal"] == 2:
        mid_x = int((terminal_detections[0]["center_2d"][0] + terminal_detections[1]["center_2d"][0])/2)
        mid_y = int((terminal_detections[0]["center_2d"][1] + terminal_detections[1]["center_2d"][1])/2)
    depth = depth_frame.get_distance(mid_x, mid_y)

    # 3D point
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [mid_x, mid_y], depth)
        # Check if there is only 1 bbox for class 0 (battery_housing)
        # Check closest to class 0 (optional)
        # Check if there is only 2 bbox for class 1
        # Take coordinate of these two bounding box and calculate center of connecting line
        # generate 3D coordinate on this point
        # transform from camera frame to gripper frame
    pass




def colorize_depth(depth_frame, depth_scale, min_depth=0.2, max_depth=2.0):

    depth_image = np.asanyarray(depth_frame.get_data())

    depth_m = depth_image * depth_scale
    depth_clipped = np.clip(depth_m, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


def draw_detection(color_image, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = str(det["class_id"])
        cls_name = det["class_name"]
        conf = det["confidence"]
        xyz = det["xyz"]
        xyz_gripper = det["xyz_gripper_frame"]

        label = f"ID:{cls_id} {conf:.2f} | {cls_name}"
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Optional: show XYZ below the box
        cv2.putText(color_image, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}", 
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.putText(color_image, f"X_f:{xyz_gripper[0]:.2f} Y_f:{xyz_gripper[1]:.2f} Z_f:{xyz_gripper[2]:.2f}", 
                    (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(color_image, (640-300, 360-200), (640+300, 360+200), (0, 0, 255), 2)
    return color_image

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


if __name__ == "__main__":
    model = YOLO(r"..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")