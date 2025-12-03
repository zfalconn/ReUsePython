from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import logging
import torch
import math
# import sys
# import os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.visual_controller import tf_camera_to_gripper


def detection_xyz(model: YOLO, color_frame, depth_frame, intrinsics, img_width, img_height, **yolo_args):

    color_image = np.asanyarray(color_frame.get_data())
    with torch.inference_mode():
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

            # # Angle from opencv
            # roi = color_image[y1:y2, x1:x2]
            # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # # _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # edges = cv2.Canny(gray, 50, 150)
            # mask = cv2.dilate(edges, (3,3), iterations=1)
            
            # angle = get_object_angle(mask)

            # Depth
            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue
            
            # 3D point
            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
            #logging.info(f"POINT 3D:  {point_3d} --- {cls}")

            point_3d_gripper = tf_camera_to_gripper(point_3d, 
                                                    t_gc=np.array([0.0, 0.0, 0.0]))
            logging.info(f"POINT 3D in gripper's frame: {point_3d_gripper} --- {cls}")
           
            detections.append({
                "class_id": cls,
                "class_name": model.names[cls],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2], #pixel
                "center_2d": [cx, cy], #pixel 
                "xyz": point_3d , #m
                "xyz_gripper_frame": point_3d_gripper, #m
            })

    return detections


def detection_xyz_obb(
    model: YOLO,
    color_frame,
    depth_frame,
    intrinsics,
    img_width,
    img_height,
    **yolo_args
):
    color_image = np.asanyarray(color_frame.get_data())

    with torch.inference_mode():
        results = model(color_image, **yolo_args)[0]

    detections = []

    if not results.obb:
        return detections

    for box in results.obb:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # --- OBB output directly from YOLO ---
        cx, cy, w, h, angle = box.xywhr[0]
        # Clamp to image limits
        cx = int(max(0, min(cx, img_width - 1)))
        cy = int(max(0, min(cy, img_height - 1)))

        w = float(w)
        h = float(h)

        # Convert OBB to standard AABB for depth and quick checks
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # Depth
        depth = depth_frame.get_distance(cx, cy)
        if depth <= 0:
            continue

        # 3D position (camera frame)
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

        # Transform
        point_3d_gripper = tf_camera_to_gripper(point_3d, t_gc=np.array([0, 0, 0]))

        detections.append({
            "class_id": cls,
            "class_name": model.names[cls],
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],   # for visual reference
            "obb": [cx, cy, w, h, angle],
            "center_2d": [cx, cy],
            "xyz": point_3d,
            "xyz_gripper_frame": point_3d_gripper,
            "angle": angle
        })

    return detections



def colorize_depth(depth_frame, depth_scale, min_depth=0.2, max_depth=2.0):

    depth_image = np.asanyarray(depth_frame.get_data())

    depth_m = depth_image * depth_scale
    depth_clipped = np.clip(depth_m, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


def draw_detection(color_image, detections, limit_box = True):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center_2d"]
        cls_id = str(det["class_id"])
        cls_name = det["class_name"]
        conf = det["confidence"]
        xyz = det["xyz"]
        xyz_gripper = det["xyz_gripper_frame"]
        # angle = det["angle"]

        label = f"ID:{cls_id} {conf:.2f} | {cls_name}"
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Optional: show XYZ below the box
        cv2.putText(color_image, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}", 
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.putText(color_image, f"X_f:{xyz_gripper[0]:.2f} Y_f:{xyz_gripper[1]:.2f} Z_f:{xyz_gripper[2]:.2f}", 
                    (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # #Draw angle vector
        # draw_orientation_arrow(color_image,(cx,cy), angle )
        
    
    #Limit
    if limit_box:
        cv2.rectangle(color_image, (640-400, 360-250), (640+400, 360+250), (0, 0, 255), 2)
        cv2.circle(color_image,(640,360),2,(255,255,255))    
    return color_image


def draw_detection_obb(color_image, detections, limit_box=True):
    for det in detections:

        # --- Extract core fields ---
        cx, cy, w, h, angle_rad = det["obb"]     # YOLO11-OBB gives angle in radians (CCW)
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = float(det["confidence"])
        xyz = det["xyz"]
        xyz_gripper = det["xyz_gripper_frame"]

        # --- Build label text ---
        label = f"ID:{cls_id} {conf:.2f} | {cls_name}"

        # --- Convert angle for OpenCV ---
        # YOLO angle:  radians CCW
        # OpenCV:      degrees CW
        angle_deg = (angle_rad * 180.0 / math.pi)

        # --- Create rotated rectangle ---
        rrect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))

        # Get rotated box polygon
        box = cv2.boxPoints(rrect)      # (4,2)
        box = np.int32(box)

        # --- Draw OBB polygon ---
        cv2.polylines(color_image, [box], True, (0, 255, 0), 2)

        # --- Choose stable top-left corner for labeling ---
        # smallest x+y gives top-left orientation across rotations
        tl_idx = np.argmin(box[:, 0] + box[:, 1])
        tl = box[tl_idx]

        # --- Draw label ---
        cv2.putText(
            color_image,
            label,
            (tl[0], tl[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

        # --- Draw depth + 3D info ---
        base_x, base_y = tl[0], tl[1] + 15

        cv2.putText(
            color_image,
            f"X:{xyz[0]:.3f} Y:{xyz[1]:.3f} Z:{xyz[2]:.3f}",
            (base_x, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1
        )

        cv2.putText(
            color_image,
            f"Xf:{xyz_gripper[0]:.3f} Yf:{xyz_gripper[1]:.3f} Zf:{xyz_gripper[2]:.3f}",
            (base_x, base_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1
        )

    # --- Draw limit box if enabled ---
    if limit_box:
        cv2.rectangle(color_image, (640 - 400, 360 - 250), (640 + 400, 360 + 250), (0, 0, 255), 2)
        cv2.circle(color_image, (640, 360), 2, (255, 255, 255), -1)

    return color_image


# def tf_camera_to_gripper(point_cam, 
#                         R_gc = np.array([
#                             [0, -1, 0],
#                             [0,  0, 1],
#                             [-1, 0, 0]
#                         ]), 
#                         t_gc = np.array([0.085, -0.220, 0.040])):
#     """
#     Transform a 3D point from the camera frame to the gripper frame.

#     Args:
#         point_cam (array-like): [x, y, z] in camera frame
#         R_gc (np.ndarray): 3x3 rotation matrix from camera to gripper
#         t_gc (array-like): 3x1 translation vector from camera to gripper

#     Returns:
#         np.ndarray: [x, y, z] in gripper frame
#     """
#     point_cam = np.array(point_cam).reshape(3, 1)
#     R_gc = np.array(R_gc).reshape(3, 3)
#     t_gc = np.array(t_gc).reshape(3, 1)

#     point_gripper = R_gc @ point_cam + t_gc
#     return point_gripper.flatten()

def get_object_angle(mask: np.ndarray) -> float:
    """
    Compute the orientation angle of an object from its binary mask.
    
    Args:
        mask (np.ndarray): Binary image (0 or 255) where the object is white.
    
    Returns:
        float: Angle in degrees. 0° = horizontal, increasing counterclockwise.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour (assuming main object)
    contour = max(contours, key=cv2.contourArea)

    # Fit a rotated rectangle
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]

    # Normalize OpenCV angle convention
    # cv2.minAreaRect returns angle in range [-90, 0)
    if angle < -45:
        angle = 90 + angle

    return angle

def draw_orientation_arrow(image: np.ndarray, rect_center: tuple, angle: float, length: int = 50):
    """
    Draws an arrow indicating object orientation.

    Args:
        image (np.ndarray): The original image.
        rect_center (tuple): (x, y) center of the object.
        angle (float): Orientation angle in degrees.
        length (int): Length of the arrow to draw.
    """
    x, y = rect_center
    # Convert angle to radians
    theta = np.deg2rad(angle)

    # Compute arrow end point
    x2 = int(x + length * np.cos(theta))
    y2 = int(y - length * np.sin(theta))  # y-axis is inverted in images

    # Draw arrow
    cv2.arrowedLine(image, (int(x), int(y)), (x2, y2), (0, 0, 255), 2, tipLength=0.3)

    # Draw angle label
    cv2.putText(image, f"{angle:.1f}°", (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    model = YOLO(r"..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")