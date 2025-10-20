import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import json
import pandas as pd
from math import radians

# ---------- USER PARAMETERS ----------
image_folder = "data/calib_images_charuco"
pose_file = "data/robot/poses_xyzrxryrz.csv"

# --- ChArUco board configuration ---
squares_x = 8           # number of squares along X
squares_y = 6           # number of squares along Y
square_length = 0.020   # in meters (e.g. 20 mm)
marker_length = 0.015   # in meters (e.g. 15 mm)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

# ---------- LOAD ROBOT POSES ----------
df = pd.read_csv(pose_file, sep=r'\s+|,', engine='python')
robot_poses = []

for _, row in df.iterrows():
    x, y, z = row['X'], row['Y'], row['Z']
    rx, ry, rz = row['RX'], row['RY'], row['RZ']

    # Convert Euler angles (deg → rad) and build rotation matrix
    rvec_rad = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rvec_rad[0]), -np.sin(rvec_rad[0])],
                   [0, np.sin(rvec_rad[0]), np.cos(rvec_rad[0])]])
    Ry = np.array([[np.cos(rvec_rad[1]), 0, np.sin(rvec_rad[1])],
                   [0, 1, 0],
                   [-np.sin(rvec_rad[1]), 0, np.cos(rvec_rad[1])]])
    Rz = np.array([[np.cos(rvec_rad[2]), -np.sin(rvec_rad[2]), 0],
                   [np.sin(rvec_rad[2]), np.cos(rvec_rad[2]), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    t = np.array([[x], [y], [z]]) / 1000.0  # mm → meters
    robot_poses.append((R, t))

print(f"Loaded {len(robot_poses)} robot poses from file.")

# ---------- LOAD IMAGES ----------
image_files = sorted(glob.glob(f"{image_folder}/*.png"))
print(f"Found {len(image_files)} images.")

# ---------- DETECT CHARUCO CORNERS ----------
all_corners, all_ids, all_imgs = [], [], []

for fname in image_files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    # print(corners)
    # print(ids)
    # visualize detections
    vis = aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imshow("Aruco Detection", vis)
    cv2.waitKey(100)

    print(f"Detected {len(corners)} markers")

    if ids is not None and len(ids) > 0:

        _, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        print(ch_corners)
        if ch_corners is not None and ch_ids is not None and len(ch_corners) > 2:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            all_imgs.append(gray)
            print(f"Detected ChArUco board in {fname}")
        else:
            print(f"⚠️ Not enough ChArUco corners in {fname}")
    else:
        print(f"⚠️ No ArUco markers found in {fname}")

print(f"\nDetected ChArUco board in {len(all_corners)} / {len(image_files)} images.")

# ---------- CALIBRATE CAMERA ----------
if len(all_corners) < 3:
    raise RuntimeError("Not enough valid images for calibration.")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=all_imgs[0].shape[::-1],
    cameraMatrix=np.array([[911.6036987304688, 0.0, 640.2818603515625],
              [0.0, 911.7769775390625, 387.0967102050781],
              [0.0, 0.0, 1.0]]),
    distCoeffs=np.array([[ 0.09014592, -0.23150788, 0.0007583, -0.00150855, 0.0947352]])
)

print("\n=== Camera Intrinsics ===")
print("RMS Error:", ret)
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs.ravel())

# ---------- EXTRACT CAMERA→CHARUCO POSES ----------
R_target2cam, t_target2cam = [], []

for ch_corners, ch_ids, gray in zip(all_corners, all_ids, all_imgs):
    retval, rvec, tvec = aruco.estimatePoseCharucoBoard(
        ch_corners, ch_ids, board, camera_matrix, dist_coeffs, None, None
    )
    if retval:
        R, _ = cv2.Rodrigues(rvec)
        R_target2cam.append(R)
        t_target2cam.append(tvec)

print(f"Estimated {len(R_target2cam)} camera poses.")

# ---------- MATCH ROBOT POSES ----------
R_gripper2base = []
t_gripper2base = []
for R, t in robot_poses[:len(R_target2cam)]:
    R_gripper2base.append(R)
    t_gripper2base.append(t)

# ---------- HAND–EYE CALIBRATION ----------
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("\n=== Hand–Eye Calibration Result ===")
print("R_cam2gripper:\n", R_cam2gripper)
print("t_cam2gripper (m):\n", t_cam2gripper.ravel())

# ---------- SAVE RESULTS ----------
result = {
    "R_cam2gripper": R_cam2gripper.tolist(),
    "t_cam2gripper": t_cam2gripper.tolist(),
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.tolist()
}

with open("handeye_charuco_result.json", "w") as f:
    json.dump(result, f, indent=4)

print("\nSaved hand-eye calibration to handeye_charuco_result.json")
