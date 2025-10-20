import pyrealsense2 as rs
import cv2
import numpy as np

def realsense_init(width = 1280, height = 720, fps = 15, enable_imu = False) -> rs.pipeline:
    """
    Initialize RealSense pipeline with params

    Args:
        width (int): Frame width
        height (int): Frame height
        fps (int): (Max) Frame rate
        enable_imu (bool): Enable accel and gyro streams
    Returns:
        pipeline: Started RealSense pipeline
    """

    pipeline = rs.pipeline()
    config = rs.config()

    # Depth
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) 
    
    # Color
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Accel & Gyro
    if enable_imu:
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return pipeline, depth_scale




def realsense_get_frame(pipeline):
    """
    Get frame from Intel Realsense camera via started pipeline object

    Args:
        pipeline (rs.pipeline) : started pipeline

    Returns:
        depth_Frame (np.ndarray) : depth frame of shape (H,W) uint16
        color_frame (np.ndarray) : RGB color frame of shape (H,W,3) uint8
    """

    #Grab frame
    frames = pipeline.wait_for_frames()

    #Get color and depth frames from frame
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None

    color_frame = np.asanyarray(color_frame.get_data())
    #depth_frame = np.asanyarray(depth_frame.get_data())


    return color_frame, depth_frame