import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ---- Settings ----
output_folder = "data/yolo_retrain_251017"
os.makedirs(output_folder, exist_ok=True)

pose_counter = 1
padding = 5  # number of digits for zero padding, e.g., 3 -> 001, 002

# ---- Configure RealSense pipeline ----
pipeline = rs.pipeline()
config = rs.config()

# D435i default resolution: 1280x720 or 640x480
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

print("✅ Press SPACE to save an image with current pose number.")
print("❌ Press ESC to quit.")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Display the live feed
        cv2.imshow("D435i Capture", color_image)
        key = cv2.waitKey(1) & 0xFF

        # Save image on SPACE press
        if key == 32:  # SPACE key
            filename = os.path.join(output_folder, f"morrow{pose_counter:0{padding}d}.png")
            cv2.imwrite(filename, color_image)
            print(f"💾 Saved {filename}")
            pose_counter += 1

        # Quit on ESC
        elif key == 27:  # ESC key
            print("👋 Exiting...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
