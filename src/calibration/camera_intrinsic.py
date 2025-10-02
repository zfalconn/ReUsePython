import pyrealsense2 as rs
import numpy as np
import cv2
import json

# -------- Parameters --------
checkerboard_size = (7, 5)   # inner corners (columns, rows)
square_size = 0.020         # size of each square in meters

# Prepare object points
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

# -------- Start RealSense pipeline --------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Instructions:")
print(" - Press SPACE to capture current frame for calibration")
print(" - Move the robot to a new pose between captures")
print(" - Press ESC to finish capturing and calibrate")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find checkerboard
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)

        cv2.imshow("Calibration Capture", img)
        key = cv2.waitKey(1)

        if key & 0xFF == 27:  # ESC key
            break
        elif key & 0xFF == 32:  # SPACE key
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"Captured {len(objpoints)} frames")
            else:
                print("Checkerboard not detected, try again")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# -------- Calibrate Camera --------
if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("Calibration RMS error:", ret)
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # Save calibration
    calib_data = {
        "rms_error": float(ret),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": gray.shape[::-1]
    }

    with open("d435i_calibration.json", "w") as f:
        json.dump(calib_data, f, indent=4)

    print("Calibration saved to d435i_calibration.json")
else:
    print("No valid frames captured, calibration aborted.")
