# handeye_pipeline.py
import os
import glob
import cv2
import numpy as np
import pandas as pd

# ---------------------------
# CONFIG / ASSUMPTIONS
# ---------------------------
poses_csv = "data/robot/poses.csv"          # CSV with columns: Pose,S,L,U,R,B,T
images_folder = "data/calib_images"         # images named pose01.png, pose02.png, ...
pattern_size = (7, 5)            # inner corners (cols, rows) - change if needed
square_size = 0.020              # meters (25 mm squares) - change if needed

# Camera intrinsics (replace with your calibrated values if different)
K = np.array([[911.6036987304688, 0.0, 640.2818603515625],
              [0.0, 911.7769775390625, 387.0967102050781],
              [0.0, 0.0, 1.0]])
dist_coeffs = np.array([[ 0.03760835, 0.24577186, -0.005007, -0.00701297, -1.04483411]])  # replace if you have distortion coeffs

# ---------------------------
# DH PARAMETERS (from user)
# Table columns: i, theta_i(var) (this is the theta offset), d_i (const), a_i (const), alpha_i (const)
# Provided values in degrees for alpha and theta offsets; distances in mm.
# We'll convert to radians and meters.
# ---------------------------
dh_table = [
    # (theta_offset_deg, d_mm, a_mm, alpha_deg)
    (0.0,     0.0,   0.0,  -90.0),   # joint 1
    (-90.0,   0.0, 700.0, 180.0),    # joint 2
    (0.0,     0.0,   0.0,   90.0),   # joint 3
    (0.0, -500.0,   0.0,  -90.0),    # joint 4
    (0.0,  -162.0,  0.0,   90.0),    # joint 5
    (0.0,  -170.0,  0.0,    0.0),    # joint 6
]

# Convert DH numeric units
dh_params = []
for theta_off_deg, d_mm, a_mm, alpha_deg in dh_table:
    dh_params.append({
        "theta_offset": np.deg2rad(theta_off_deg),
        "d": d_mm / 1000.0,          # meters
        "a": a_mm / 1000.0,          # meters
        "alpha": np.deg2rad(alpha_deg)
    })

# ---------------------------
# Helpers: DH transform, FK
# ---------------------------
def dh_transform(a, alpha, d, theta):
    """Return 4x4 DH homogeneous transform for given params (all in SI units, theta in radians)."""
    ca = np.cos(alpha); sa = np.sin(alpha)
    ct = np.cos(theta); st = np.sin(theta)
    T = np.array([
        [ ct, -st*ca,  st*sa, a*ct],
        [ st,  ct*ca, -ct*sa, a*st],
        [  0,     sa,     ca,    d],
        [  0,      0,      0,    1]
    ])
    return T

def forward_kinematics_from_joints(joint_angles_deg):
    """
    joint_angles_deg: list-like of 6 joint angles as given by your robot (in degrees)
    returns T_base_gripper (4x4)
    """
    if len(joint_angles_deg) != 6:
        raise ValueError("Expected 6 joint angles.")

    T = np.eye(4)
    for i, q_deg in enumerate(joint_angles_deg):
        q = np.deg2rad(q_deg)  # convert joint reading to radians
        params = dh_params[i]
        theta = q + params["theta_offset"]
        Ti = dh_transform(params["a"], params["alpha"], params["d"], theta)
        T = T @ Ti
    return T

# ---------------------------
# Load poses CSV and images
# ---------------------------
df = pd.read_csv(poses_csv)
image_files = sorted(glob.glob(os.path.join(images_folder, "pose*.png")))
if len(image_files) != len(df):
    print(f"Warning: number of images ({len(image_files)}) != number of poses ({len(df)}).")

# ---------------------------
# Prepare object points for checkerboard
# ---------------------------
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
# create grid in (x,y)
xs, ys = np.indices(pattern_size)
objp[:, :2] = np.stack([xs.ravel(), ys.ravel()], axis=1) * square_size

# ---------------------------
# Loop: compute FK and solvePnP for each image
# ---------------------------
R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list = []
t_target2cam_list = []

for idx, row in df.iterrows():
    pose_idx = int(row["Pose"]) if "Pose" in row.index else (idx+1)
    # read joints in the order S,L,U,R,B,T
    joints = [row["S"], row["L"], row["U"], row["R"], row["B"], row["T"]]
    T_base_gripper = forward_kinematics_from_joints(joints)
    R_g2b = T_base_gripper[:3, :3]
    t_g2b = T_base_gripper[:3, 3].reshape(3, 1)

    R_gripper2base_list.append(R_g2b)
    t_gripper2base_list.append(t_g2b)

    # load corresponding image (try to match by index)
    if idx < len(image_files):
        img = cv2.imread(image_files[idx])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"Checkerboard not found in image {image_files[idx]} (pose {pose_idx}).")
            continue

        # refine corners
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)

        # solvePnP
        ok, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print(f"solvePnP failed for image {image_files[idx]}")
            continue

        R_t2c, _ = cv2.Rodrigues(rvec)         # rotation matrix target->camera
        t_t2c = tvec.reshape(3,1)
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(t_t2c)
    else:
        print(f"No image for pose row {idx+1}; skipping solvePnP for this pose.")

# Check we have enough pairs
n_pairs = min(len(R_gripper2base_list), len(R_target2cam_list))
print(f"Collected {len(R_gripper2base_list)} robot poses and {len(R_target2cam_list)} checkerboard poses.")
if n_pairs < 3:
    raise RuntimeError("Need at least 3 paired poses (robot + checkerboard).")

# Trim lists to same length (pairwise correspond)
R_gripper2base_list = R_gripper2base_list[:n_pairs]
t_gripper2base_list = t_gripper2base_list[:n_pairs]
R_target2cam_list = R_target2cam_list[:n_pairs]
t_target2cam_list = t_target2cam_list[:n_pairs]

# ---------------------------
# Run OpenCV hand-eye calibration
# ---------------------------
# OpenCV expects lists of 3x3 rotations (numpy arrays) and 3x1 translations
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base_list, t_gripper2base_list,
    R_target2cam_list, t_target2cam_list,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("\n=== Hand-Eye result: camera w.r.t gripper ===")
print("R_cam2gripper:\n", R_cam2gripper)
print("t_cam2gripper (meters):\n", t_cam2gripper.ravel())

# Save outputs
np.save("R_gripper2base_list.npy", np.array(R_gripper2base_list))
np.save("t_gripper2base_list.npy", np.array(t_gripper2base_list))
np.save("R_target2cam_list.npy", np.array(R_target2cam_list))
np.save("t_target2cam_list.npy", np.array(t_target2cam_list))
np.save("R_cam2gripper.npy", R_cam2gripper)
np.save("t_cam2gripper.npy", t_cam2gripper)

print("Saved results to disk.")
