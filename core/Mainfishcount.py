#!/usr/bin/env python3
import cv2
import threading
import datetime
import time
import numpy as np
import os
import queue
from ultralytics import YOLO
# from core.sort import Sort (Removed in favor of ByteTrack)
from openpyxl import Workbook, load_workbook
import cvzone
from collections import defaultdict

MODEL_PATH = "weights/best+.pt" # Modèle PyTorch Original
VIDEO_PATH = "data/dorada9.mp4" 
OUTPUT_DIR = "results"
REPORT_DIR = "results"
TRACKER_CFG = "core/custom_bytetrack.yaml"

# Visual Constants
COLOR_VIOLET = (255, 0, 255) # Pink/Violet as requested
COLOR_ACCENT = (0, 165, 255) # Golden-Orange for Branding
HEADER_H = 60
QUEUE_MAXSIZE = 30 
FONT = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(OUTPUT_DIR, exist_ok=True)

class SingleCamFishCounter:
    def __init__(self, model_path, video_source):
        self.model = YOLO(model_path, task='detect')
        self.video_source = video_source
        self.frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

        self.counter = set()
        self.object_count = 0
        self.track_history = defaultdict(lambda: [])

        self.lock = threading.Lock()
        self.new_frame_event = threading.Event() # For display sync
        self.frame = None
        self.stopped = False
        self.processed_frames = 0

        self.cap = cv2.VideoCapture(video_source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # Vitesse d'export normale (Le ralenti sera généré dynamiquement image par image)
        export_fps = self.fps
        
        # Professional Video Writer (MP4 format)
        self.writer = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, f"AquaVision_Report_{self.timestamp}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), 
            export_fps,
            (960, 540)
        )
        
        # CSV Logging Initialization
        self.csv_path = os.path.join(REPORT_DIR, f"DataLog_{self.timestamp}.csv")
        with open(self.csv_path, 'w') as f:
            f.write("Timestamp,Frame,Total_Fish,Status\n")

    def capture_thread(self):
        # Ralentissement renforcé (1.7x) pour une fluidité totale
        delay = 1.7 / self.fps 
        
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] Fin de la vidéo.")
                break

            # Pause prolongée pour une analyse visuelle confortable
            time.sleep(delay)

            # --- FRAME DROP LOGIC (ANTI-LAG) ---
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
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

            resized = cv2.resize(frame, (960, 540))

            with self.lock:
                self.frame = resized
                self.new_frame_event.set() # Notify display thread

            # --- Effet "Time-Ramping" (Ralenti Cinématique Extrême) ---
            # Uniquement pour le fichier MP4 (L'affichage en direct reste parfaitement normal)
            num_writes = 1
            if self.processed_frames < self.fps * 5:    # Phase 1: De 0 à 5 secondes
                num_writes = 8  # Ultra-Ralenti EXTREME (0.12x la vitesse normale)
            elif self.processed_frames < self.fps * 10: # Phase 2: De 5 à 10 secondes
                num_writes = 4  # Ralenti classique (0.25x)
            else:
                num_writes = 1  # Phase 3: Vitesse Classique (1.0x) pour le reste
                
            for _ in range(num_writes):
                self.writer.write(resized)

            self.processed_frames += 1

    def draw_aqua_header(self, frame):
        """Draws the translucent black header and the AquaVision branding."""
        overlay = frame.copy()
        # 1. Barre noire translucide
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], HEADER_H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 2. Branding AquaVision sur la droite
        cv2.putText(frame, "AQUAVISION | SEABREAM AI", (frame.shape[1]-400, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_ACCENT, 2)

    def process_frame(self, frame):
        # 1. Inference with ByteTrack (High Sensitivity: 0.1)
        results = self.model.track(
            frame, 
            persist=True, 
            device='cpu', 
            tracker=TRACKER_CFG,
            conf=0.01,   # Confiance quasi-nulle pour détecter très loin
            iou=0.45,    # (0.45) Le bon réglage pour éviter d'avoir 2 boîtes sur le même poisson
            imgsz=1024,  
            verbose=False
        )

        # 2. VIOLET DRAWING - Back to original style
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id
            
            if ids is not None:
                ids = ids.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Counting
                    if track_id not in self.counter:
                        self.counter.add(track_id)
                        with open(self.csv_path, 'a') as f:
                            f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')},{self.processed_frames},{len(self.counter)},TRACKING\n")
                    
                    # Track Trail (Petite ligne pour le suivi)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    track = self.track_history[track_id]
                    track.append((cx, cy))
                    if len(track) > 8:  # Ligne très courte (environ la longueur de la dorade)
                        track.pop(0)

                    # 1. Trajectoire "Ultra-Courte" (La toute petite queue rapide)
                    if len(track) > 3:  # Disparaît instantanément (seulement les 3 dernières positions)
                        track.pop(0)

                    try:
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=COLOR_VIOLET, thickness=2)
                    except Exception:
                        pass
                    
                    # 2. Centroïde ("Point Central Blanc" pour cibler)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

                    # 3. Retour aux Rectangles Violets Pro et IDs clairs avec fond (comme demandé)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VIOLET, 2)
                    cvzone.putTextRect(frame, f'ID {int(track_id)}', [max(0, x1), max(10, y1-10)], 
                                     scale=1, thickness=2, colorR=COLOR_VIOLET)
        self.object_count = len(self.counter)
        
        # 3. Bandeau Noir AquaVision
        self.draw_aqua_header(frame)

        # 4. Violet Counter in Top-Left (Original Style)
        cvzone.putTextRect(frame, f'Total Fish = {self.object_count}', [50, 50], 
                         scale=2, thickness=3, colorR=COLOR_VIOLET)

        return frame

    def display_thread(self):
        cv2.namedWindow("Fish Counter - Master Edition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fish Counter - Master Edition", 1280, 720)

        while not self.stopped:
            if self.new_frame_event.wait(timeout=0.01):
                self.new_frame_event.clear()
                
                with self.lock:
                    if self.frame is not None:
                        frame = self.frame.copy()
                        cv2.imshow("Fish Counter - Master Edition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
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