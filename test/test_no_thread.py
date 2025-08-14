import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time

# === Initialize RealSense Pipeline ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale:", depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# === Load YOLOv8 model ===
model = YOLO(r"..\models\original\Focus1_YOLO11n_x1024_14112024.pt")

try:
    frame_count = 0
    start_time = time.time()

    total_latency = 0.0  # total iteration time accumulator
    avg_latency = 0.0    # initialize rolling average

    while True:
        iter_start = time.perf_counter()  # high-res timer

        # Wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Inference
        results = model(color_image, conf=0.7, device='cpu')[0]

        # Compute overall FPS
        frame_count += 1
        elapsed_time = time.perf_counter() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))

            depth = depth_frame.get_distance(cx, cy)
            intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

            label = f"{model.names[cls]} {depth:.2f}m"
            print(f"[{model.names[cls]}] at ({cx}, {cy}) → {depth:.2f}m → 3D: {np.round(point_3d, 2)}")

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("YOLOv8 + RealSense", color_image)

        # === Timing ===
        iter_end = time.perf_counter()
        iteration_duration = iter_end - iter_start  # per-frame latency
        total_latency += iteration_duration

        # Rolling average (EMA)
        alpha = 0.1
        avg_latency = alpha * iteration_duration + (1 - alpha) * avg_latency

        print(f"[Frame {frame_count}] Current: {iteration_duration*1000:.2f} ms | Avg: {avg_latency*1000:.2f} ms | FPS: {fps:.2f}")

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
