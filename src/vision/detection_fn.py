from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
import logging
import torch
import math

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

if __name__ == "__main__":
    model = YOLO(r"..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")