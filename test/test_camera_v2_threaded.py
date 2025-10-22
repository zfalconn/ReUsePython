import sys, os, cv2, time, threading, queue
import numpy as np
from ultralytics import YOLO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.detection_fn import detection_xyz, draw_detections


def colorize_depth(depth_image, depth_scale, min_depth=0.2, max_depth=2.0):
    depth_m = depth_image * depth_scale
    depth_clipped = np.clip(depth_m, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


def detection_worker(frame_queue, output_queue, model, camera, running_flag):
    """Thread: YOLO inference."""
    while running_flag["run"]:
        try:
            color_frame, depth_frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        detections = detection_xyz(model, color_image, depth_frame, conf=0.75)
        color_annotated = draw_detections(color_image, detections)

        depth_colored = colorize_depth(depth_image, camera.get_depth_scale())
        output_queue.put((color_annotated, depth_colored))

    print("Detection thread exiting.")

def display_worker(output_queue, running_flag):
    color_window = "Color Stream"
    depth_window = "Depth Stream"

    # Create resizable windows
    cv2.namedWindow(color_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)

    # Set initial size
    cv2.resizeWindow(color_window, 1280, 720)
    cv2.resizeWindow(depth_window, 1280, 720)

    while running_flag["run"]:
        try:
            color_image, depth_image = output_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Display each image in its own window
        cv2.imshow(color_window, color_image)
        cv2.imshow(depth_window, depth_image)

        # ESC to quit
        if cv2.waitKey(1) == 27:
            running_flag["run"] = False
            break

    cv2.destroyAllWindows()




if __name__ == "__main__":
    camera = RealSenseStream(fps=30, width=1280, height=720)
    model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")

    frame_queue = queue.Queue(maxsize=2)   # Small buffer to avoid lag
    output_queue = queue.Queue(maxsize=2)
    running_flag = {"run": True}

    # Start camera acquisition
    camera.start()

    def frame_reader():
        """Pull frames from camera and push to queue (producer)."""
        while running_flag["run"]:
            frames = camera.get_latest_frame()
            if frames:
                if frame_queue.full():
                    _ = frame_queue.get_nowait()  # Drop old frame
                frame_queue.put(frames)
            time.sleep(0.01)

    # Threads
    t_reader = threading.Thread(target=frame_reader, daemon=True)
    t_detector = threading.Thread(target=detection_worker, args=(frame_queue, output_queue, model, camera, running_flag), daemon=True)
    t_display = threading.Thread(target=display_worker, args=(output_queue, running_flag), daemon=True)

    # Start
    t_reader.start()
    t_detector.start()
    t_display.start()

    try:
        while running_flag["run"]:
            time.sleep(0.5)
    except KeyboardInterrupt:
        running_flag["run"] = False

    camera.stop()
    cv2.destroyAllWindows()
    print("✅ All threads stopped.")
