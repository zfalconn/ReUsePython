import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ---- Settings ----
output_folder = "data/calib_images"
os.makedirs(output_folder, exist_ok=True)

pose_counter = 1

# ---- Configure RealSense pipeline ----
pipeline = rs.pipeline()
config = rs.config()

# D435i default resolution: 1280x720 or 640x480
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

print("Press SPACE to save an image with current pose number.")
print("Press ESC to quit.")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Show live stream
        cv2.imshow("D435i Capture", color_image)
        key = cv2.waitKey(1) & 0xFF

        # Save image when SPACE pressed
        if key == 32:  # SPACE
            filename = os.path.join(output_folder, f"pose{pose_counter:02d}.png")
            cv2.imwrite(filename, color_image)
            print(f"Saved {filename}")
            pose_counter += 1

        # Quit on ESC
        if key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
