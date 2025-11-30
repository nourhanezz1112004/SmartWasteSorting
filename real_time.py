"""Real-time camera loop that uses inference.predict(frame)."""
import cv2
import threading
from src import inference

class CameraWorker(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.running = False
        self.frame = None
        self.pred = ('', 0.0)

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame = frame
            try:
                label, prob = inference.predict(frame)
                self.pred = (label, prob)
            except Exception:
                self.pred = ('Model not loaded', 0.0)
        self.cap.release()

    def stop(self):
        self.running = False
        self.join()
