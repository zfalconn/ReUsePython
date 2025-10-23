from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import numpy as np
# import sys
# import os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def detection_xyz(model: YOLO, color_frame, depth_frame, intrinsics, img_width, img_height, **yolo_args):

    color_image = np.asanyarray(color_frame.get_data())

    results = model(color_image, **yolo_args)[0]
    detections = []
    
    # intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

    # width = depth_frame.get_width()
    # height = depth_frame.get_height()

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
        print("POINT 3D: ", point_3d)
        detections.append({
            "class_id": cls,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "center_2d": [cx, cy],
            "xyz": point_3d  # optional: convert to list
        })

    return detections


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
        conf = det["confidence"]
        xyz = det["xyz"]

        label = f"ID:{cls_id} {conf:.2f}"
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Optional: show XYZ below the box
        cv2.putText(color_image, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}", 
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return color_image

if __name__ == "__main__":
    model = YOLO(r"..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")