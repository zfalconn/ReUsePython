import pyrealsense2 as rs
import numpy as np
import cv2

def play_realsense(file_path, show_color=True, show_depth=True, playback_speed=1.0):
    """
    Play a recorded RealSense .bag file with optional color and depth display.

    Args:
        file_path (str): Path to the .bag file.
        show_color (bool): Show color stream if available.
        show_depth (bool): Show depth stream if available.
        playback_speed (float): Speed multiplier (1.0 = normal speed).
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable file playback
    rs.config.enable_device_from_file(config, file_path, repeat_playback=False)

    # Start pipeline
    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()

    # Get playback handle to control playback behavior
    playback = device.as_playback()
    playback.set_real_time(False)  # Let us control the playback speed manually

    align = rs.align(rs.stream.color)

    print(f"▶️ Playing file: {file_path}")
    print(f"  Showing color: {show_color}, depth: {show_depth}, speed: {playback_speed}x")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            frames = align.process(frames)

            color_image, depth_image = None, None

            if show_color and frames.get_color_frame():
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())

            if show_depth and frames.get_depth_frame():
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

            # Combine views
            if show_color and show_depth and color_image is not None and depth_image is not None:
                combined = np.hstack((color_image, depth_colormap))
                cv2.imshow("Color + Depth", combined)
            elif show_color and color_image is not None:
                cv2.imshow("Color", color_image)
            elif show_depth and depth_image is not None:
                cv2.imshow("Depth", depth_colormap)

            # Playback speed control
            key = cv2.waitKey(int(33 / playback_speed))
            if key == 27:  # ESC key
                print("⏹️ Playback stopped by user.")
                break

    except RuntimeError:
        print("✅ End of recording reached.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    play_realsense(
        file_path="data/video/realsense_test_color.bag",  # Path to your recorded file
        show_color=True,
        show_depth=False,
        playback_speed=1.0,             # 1.0 = real-time
    )
