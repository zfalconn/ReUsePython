from multiprocessing import Process, Queue
import cv2
import time

# --------- Process 1: Camera acquisition (Producer) ---------
def producer(q):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if not q.full():
            q.put(frame)
        time.sleep(0.03)  # simulate ~30 FPS
    cap.release()

# --------- Process 2: Image processor (Consumer) ---------
def consumer(q_in, q_out):
    while True:
        if not q_in.empty():
            frame = q_in.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            q_out.put(gray)

# --------- Process 3: Logger ---------
def logger(q):
    while True:
        if not q.empty():
            img = q.get()
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] Processed frame with shape: {img.shape}")
            # You could write to disk or a file here too

# --------- Main function ---------
if __name__ == "__main__":
    q1 = Queue(maxsize=5)  # From camera → processor
    q2 = Queue(maxsize=5)  # From processor → logger

    p1 = Process(target=producer, args=(q1,))
    p2 = Process(target=consumer, args=(q1, q2))
    p3 = Process(target=logger, args=(q2,))

    p1.start()
    p2.start()
    p3.start()

    try:
        while True:
            time.sleep(1)  # Keep main alive
    except KeyboardInterrupt:
        print("Shutting down...")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p1.join()
        p2.join()
        p3.join()
