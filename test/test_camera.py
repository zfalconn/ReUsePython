
import sys
import os
import cv2

#Add root dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.RealSenseStream import RealSenseStream

import cv2

if __name__ == "__main__":
    camera = RealSenseStream(fps=30)
    camera.start()

    while True:
        frames = camera.get_latest_frame()
        if frames is not None:
            color_frame, depth_frame = frames
            cv2.imshow("Color", color_frame)

        if cv2.waitKey(1) == 27: # ESC to exit
            break

    camera.stop()
    cv2.destroyAllWindows()