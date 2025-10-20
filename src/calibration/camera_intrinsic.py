import cv2
import numpy as np
import glob
import json
import os

# -------- Parameters --------
checkerboard_size = (7, 5)   # inner corners (cols, rows)
square_size = 0.020          # meters
images_folder = "data/calib_images"  # folder with your saved images

# Prepare object points
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

# -------- Load images --------
image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))  # adjust extension if needed

for idx, fname in enumerate(image_files):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find checkerboard
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners2)

        # Optional: visualize detection
        cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow("Checkerboard Detection", img)
        cv2.waitKey(100)  # show each detection briefly
    else:
        print(f"Checkerboard not detected in {fname}")

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
    print("No valid checkerboards detected, calibration aborted.")
