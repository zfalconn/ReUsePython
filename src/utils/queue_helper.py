from queue import Queue, Full, Empty

def put_latest(queue : Queue, item):
    try:
        queue.put_nowait(item)
    except Full:
        try:
            _ = queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            pass