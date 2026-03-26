#!/usr/bin/env python3
import cv2
import threading
import datetime
import time
import numpy as np
import os
import queue
from ultralytics import YOLO
from core.sort import Sort
from openpyxl import Workbook, load_workbook
import cvzone

# =================== Config ===================
DROP_INTERVAL = 1
QUEUE_MAXSIZE = 3000
DISPLAY_WINDOW = "Fish Counter (No Line)"

MODEL_PATH = "weights/best@.pt"
VIDEO_PATH = "data/dorada1.mp4"

start_time = time.time()

class SingleCamFishCounter:
    def __init__(self, model_path, video_source):
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

        self.counter = set()
        self.object_count = 0

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.processed_frames = 0

        self.cap = cv2.VideoCapture(video_source)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.writer = cv2.VideoWriter(
            f"results/Processed_{timestamp}.avi",
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (640, 480)
        )

    def capture_thread(self):
        while True:   # modified
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] End of video file.")
                break   # DO NOT stop here

            try:
                self.frame_queue.put(frame)
            except:
                pass

        self.cap.release()

    def process_thread(self):
        while not self.stopped or not self.frame_queue.empty():  
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                frame = self.process_frame(frame)
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

            resized = cv2.resize(frame, (640, 480))

            with self.lock:
                self.frame = resized

            self.writer.write(resized)

            self.processed_frames += 1

    def process_frame(self, frame):
        results = self.model(frame, device='cpu')
        detections = np.empty((0, 5))

        for info in results:
            for box in info.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()

                if conf > 0.30:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        tracks = self.tracker.update(detections)

        for result in tracks:
            x1, y1, x2, y2, track_id = map(int, result)

            if track_id not in self.counter:
                self.counter.add(track_id)

            cx, cy = x1 + (x2 - x1)//2, y1 + (y2 - y1)//2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.circle(frame, (cx, cy), 5, (255,0,0), -1)

            cvzone.putTextRect(frame, f'{track_id}', [x1, y1-10], scale=1, thickness=2)

        self.object_count = len(self.counter)

        cvzone.putTextRect(frame, f'Total Fish = {self.object_count}',
                           [50,50], scale=2, thickness=3)

        return frame

    def display_thread(self):
        while not self.stopped or not self.frame_queue.empty():   
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None

            if frame is not None:
                cv2.imshow(DISPLAY_WINDOW, frame)

            if cv2.waitKey(30) & 0xFF == 27:
                self.stopped = True
                break

        cv2.destroyAllWindows()

    def run(self):
        threads = [
            threading.Thread(target=self.capture_thread, daemon=True),
            threading.Thread(target=self.process_thread, daemon=True),
            threading.Thread(target=self.display_thread, daemon=True)
        ]

        for t in threads:
            t.start()

        while any(t.is_alive() for t in threads):
            time.sleep(0.1)

        self.stopped = True   

        self.writer.release()

        print("\n=== FINAL RESULT ===")
        print(f"Total Fish Detected: {len(self.counter)}")


if __name__ == "__main__":
    counter = SingleCamFishCounter(MODEL_PATH, VIDEO_PATH)
    counter.run()