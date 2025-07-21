
import sys
import os
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import realsense_get_frame, realsense_init





pipeline, depth_scale = realsense_init()

try:
    while True:
        color_img, depth_img = realsense_get_frame(pipeline)

        if color_img is None or depth_img is None:
            continue
        depth_in_meters = depth_img * depth_scale

        cv2.imshow("RealSense Live Feed", color_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Ended via keyboard interrupt...")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()