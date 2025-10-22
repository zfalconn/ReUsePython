import pyrealsense2 as rs
import numpy as np
import cv2
import time


def record_realsense(
    output_path="recording.bag",
    record_time=10,
    enable_color=True,
    enable_depth=True,
    color_resolution=(640, 480),
    depth_resolution=(640, 480),
    color_fps=30,
    depth_fps=30,
    show=True,
):
    """
    Stream from RealSense and record to .bag file with flexible options.

    Args:
        output_path (str): Path to save .bag file.
        record_time (int): Duration to record in seconds.
        enable_color (bool): Whether to enable color stream.
        enable_depth (bool): Whether to enable depth stream.
        color_resolution (tuple): (width, height) for color.
        depth_resolution (tuple): (width, height) for depth.
        color_fps (int): Frame rate for color stream.
        depth_fps (int): Frame rate for depth stream.
        show (bool): Whether to show live preview.
    """

    pipeline = rs.pipeline()
    config = rs.config()

    # Enable recording
    config.enable_record_to_file(output_path)

    # Enable streams based on user config
    if enable_depth:
        config.enable_stream(
            rs.stream.depth,
            depth_resolution[0],
            depth_resolution[1],
            rs.format.z16,
            depth_fps,
        )
    if enable_color:
        config.enable_stream(
            rs.stream.color,
            color_resolution[0],
            color_resolution[1],
            rs.format.bgr8,
            color_fps,
        )

    # Start streaming
    pipeline.start(config)

    print("🎥 RealSense recording started")
    print(f"Saving to: {output_path}")
    print(f"Depth enabled: {enable_depth}, Color enabled: {enable_color}")
    print(f"Recording for {record_time} seconds...")

    start_time = time.time()

    try:
        while time.time() - start_time < record_time:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame() if enable_color else None
            depth_frame = frames.get_depth_frame() if enable_depth else None

            # Visualization only
            if show and (color_frame or depth_frame):
                frames_to_show = []
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    frames_to_show.append(color_image)
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET,
                    )
                    frames_to_show.append(depth_colormap)

                if len(frames_to_show) > 0:
                    combined = np.hstack(frames_to_show)
                    cv2.imshow("RealSense Recorder", combined)
                    if cv2.waitKey(1) == 27:  # ESC key to stop early
                        break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("✅ Recording finished and pipeline stopped.")


if __name__ == "__main__":
    record_realsense(
        output_path="data/video/realsense_test_colorwdepth.bag",
        record_time=20,
        enable_color=True,
        enable_depth=False,   # 👈 set to True if you want depth
        color_resolution=(1280, 720),
        depth_resolution=(1280, 720),
        color_fps=30,
        depth_fps=30,
        show=True,
    )
