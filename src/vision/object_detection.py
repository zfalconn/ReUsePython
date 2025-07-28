import threading
from queue import Queue
from .detection_fn import detection_xyz, draw_detections
from ultralytics import YOLO
import cv2
import pyrealsense2 as rs

class ObjectDetection:
    def __init__(self, model : YOLO, display : bool = False, max_queue_size = 1, **yolo_args):
        self.model = model
        self.detections_queue = Queue(maxsize=max_queue_size)
        self.display = display
        self.running = False
        self.yolo_args = yolo_args
        
    
    def _detection_xyz(self, input_frame):
        while self.running:
            color_frame, depth_frame = input_frame
            if color_frame is not None and depth_frame is not None:
                results = detection_xyz(self.model, color_frame, depth_frame, **self.yolo_args)
                if not self.detections_queue.full():
                    self.detections_queue.put(results)

    
    def _display(self, input_frame):
        while self.running and self.display:
            if input_frame is not None and not self.detections_queue.empty():
                color_frame, _ = input_frame
                detections = self.get_latest_detection()
                if detections:
                    annotated_frame = color_frame.copy()
                    annotated_frame = draw_detections(annotated_frame, detections)
                    cv2.imshow("Bounding Box with XYZ", annotated_frame)
            if cv2.waitKey(1) == 27:
                self.running = False
                break
        cv2.destroyAllWindows()

    
    def start(self, input_frame):
        self.running = True
        self.det_thread = threading.Thread(target=self._detection_xyz, args=(input_frame,), daemon=True)
        self.disp_thread = threading.Thread(target=self._display, args=(input_frame,self.get_latest_detection()), daemon=True)
        
        self.det_thread.start()
        self.disp_thread.start()


    def stop(self):
        self.running = False
        self.det_thread.join()
        self.disp_thread.join()
    
    def get_latest_detection(self):
        latest_detection = None
        while not self.detections_queue.empty():
            latest_detection = self.detections_queue.get()
        return latest_detection

if __name__ == "__main__":
    pass