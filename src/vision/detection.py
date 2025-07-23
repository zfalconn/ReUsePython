from ultralytics import YOLO
import cv2
import pyrealsense2 as rs



def detection_xyz(model : YOLO , frame, confidence=0.7):
    
    """
    Perform objection detection and generate XYZ coordinates

    """
    point_3d_list = []
    results = model(frame, conf=confidence)

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0] 
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Get depth at bounding box center
        depth = frame.get_distance(cx, cy)

        # Deproject to 3D point
        intrinsics = frame.profile.as_video_stream_profile().get_intrinsics()
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

        point_3d_list.append(point_3d)
    return point_3d_list