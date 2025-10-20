import cv2
from cv2 import aruco

# Define Charuco parameters
squares_x = 7  # number of squares in X direction
squares_y = 5  # number of squares in Y direction
square_length = 0.02   # in meters
marker_length = 0.015  # in meters

# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)

# Create Charuco board
board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

# Save to file
img = board.generateImage((1000, 800))
cv2.imwrite("charuco_board.png", img)
