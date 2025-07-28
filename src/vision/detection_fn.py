from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
# import sys
# import os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def detection_xyz(model : YOLO , color_frame, depth_frame, **yolo_args):
    
    """
    Perform objection detection and generate XYZ coordinates

    """
    results = model(color_frame, **yolo_args)[0]

    detections = []
    intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0] 
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Get depth at bounding box center
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        # Clamping to prevent out of range
        cx = max(0, min(cx, width - 1))
        cy = max(0, min(cy, height - 1))
        depth = depth_frame.get_distance(cx, cy)
        
        # Deproject to 3D point
        
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

        detections.append({
            "class_id": cls,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "center_2d": [cx, cy],
            "xyz": point_3d
        })

    return detections

def draw_detections(color_frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = str(det["class_id"])
        conf = det["confidence"]
        xyz = det["xyz"]

        label = f"ID:{cls_id} {conf:.2f}"
        cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Optional: show XYZ below the box
        cv2.putText(color_frame, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}", 
                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return color_frame

if __name__ == "__main__":
    model = YOLO(r"..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")