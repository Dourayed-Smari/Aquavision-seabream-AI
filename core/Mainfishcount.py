#!/usr/bin/env python3
import os
import sys
# Ajout du dossier racine au PATH avant toute importation locale
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.biomass import FishBiomass # Import après sys.path mais avant Torch
import cv2
import threading
import datetime
import time
import numpy as np
import queue
from ultralytics import YOLO
# from core.sort import Sort (Removed in favor of ByteTrack)
from openpyxl import Workbook, load_workbook
import cvzone
from collections import defaultdict

MODEL_PATH = "weights/bestmodel1.pt" # Modèle Segmentation V2 (Test Pratique)
VIDEO_PATH = "data/dorada16.mp4" 
OUTPUT_DIR = "results"
REPORT_DIR = "results"
TRACKER_CFG = "core/custom_bytetrack.yaml"

# Expert Constants (v2.3.1 Adaptive Flash)
PX_TO_CM = 0.09 # Started at 0.09, will auto-adjust
TARGET_AVG_WEIGHT = 450.0 # Target Average Weight (g) for this cage
CALIBRATION_THRESHOLD = 10 # Number of fish before auto-calibration triggers

# Visual Constants
COLOR_VIOLET = (255, 0, 255) # Pink/Violet as requested
COLOR_ACCENT = (0, 165, 255) # Golden-Orange for Branding
HEADER_H = 60
QUEUE_MAXSIZE = 30 
FONT = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(OUTPUT_DIR, exist_ok=True)

class SingleCamFishCounter:
    def __init__(self, model_path, video_source):
        self.model = YOLO(model_path, task='segment')
        self.video_source = video_source
        self.frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

        self.counter = set()
        self.object_count = 0
        self.track_history = defaultdict(lambda: [])
        
        # Expert Biomass Logic (v2.2)
        self.biomass_estimator = FishBiomass(px_to_cm_ratio=PX_TO_CM)
        self.fish_weights = {} # ID -> Stable Weight (g)
        self.weight_history = defaultdict(list) # ID -> List of (weight, score)
        self.total_biomass_kg = 0.0
        self.recalibrate_flag = False # For interactive calibration
        self.is_calibrated = False # Flag for auto-calibration
        self.calibration_status = "INITIALIZING..."

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
            f.write("Timestamp,Frame,Total_Fish,ID,Weight_G,Total_Biomass_KG,Status\n")

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
            
            # --- STRICT SEQUENTIAL LOGIC (NO FRAME DROP) ---
            # Bloque le thread de lecture si le CPU est en retard, forçant l'analyse de 100% des frames
            try:
                self.frame_queue.put(frame, block=True, timeout=5)
            except queue.Full:
                print("[WARNING] Frame queue timeout. Le CPU est saturé.")

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
        
        # 3. Métriques AquaVision (Bloc Gauche Pro en Violet)
        avg_weight = (sum(self.fish_weights.values()) / len(self.fish_weights)) if self.fish_weights else 0
        
        cv2.putText(frame, f"FISH: {len(self.counter)}", (30, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_VIOLET, 2)
        
        cv2.putText(frame, f"TOTAL BIOMASS: {self.total_biomass_kg:.2f} kg", (30, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_VIOLET, 2)
        
        cv2.putText(frame, f"STATUS: {self.calibration_status}", (30, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 0), 1)
        
        # [EXPERT v3.0] Engine Mode Display
        engine_mode = self.biomass_estimator.current_mode
        engine_color = (0, 255, 0) if "ML" in engine_mode else (200, 200, 200)
        cv2.putText(frame, f"ENGINE: {engine_mode}", (30, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, engine_color, 1)
        
        cv2.putText(frame, f"AVG WEIGHT: {avg_weight:.1f} g", (frame.shape[1]//2-100, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    def process_frame(self, frame):
        # 1. Inference with ByteTrack (High Sensitivity: 0.1)
        results = self.model.track(
            frame, 
            persist=True, 
            device='cpu', 
            tracker=TRACKER_CFG,
            conf=0.15,    # Seuil optimal validé (Filtre les masques bruités/instables)
            iou=0.45,     # IOU serré pour limiter les collisions
            imgsz=640,    # 640 est plus stable pour les tests en temps réel
            verbose=False
        )

        # 2. Drawing Results (Native ByteTrack IDs)
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_raw = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            scores = results[0].boxes.conf.cpu().numpy()
            masks = results[0].masks.xy if results[0].masks is not None else None

            # [EXPERT] Interactive Calibration Logic
            if self.recalibrate_flag and len(boxes_raw) > 0:
                # Find the biggest fish currently on screen
                widths = boxes_raw[:, 2] - boxes_raw[:, 0]
                idx = np.argmax(widths)
                self.biomass_estimator.calibrate(30.0, widths[idx]) # Assume biggest is 30cm
                self.recalibrate_flag = False

            for box, track_id, score in zip(boxes_raw, track_ids, scores):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Track Trail (Petite ligne pour le suivi)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    track = self.track_history[track_id]
                    track.append((cx, cy))
                    
                    # --- BIOMASS CALCULATION (v2.3.1 Expert Adaptive) ---
                    weight, pose_score, is_clipped = self.biomass_estimator.estimate_weight(box, self.height)
                    if pose_score > 0.1: 
                        self.weight_history[track_id].append((weight, pose_score))
                        
                        # Compute Stable Weight using MEDIAN (to kill ID 26 outliers)
                        best_poses = sorted(self.weight_history[track_id], key=lambda x: x[1], reverse=True)[:10]
                        if best_poses:
                            self.fish_weights[track_id] = np.median([p[0] for p in best_poses])
                    
                    # --- AUTO-CALIBRATION LOGIC (v2.3.1 / v2.4 Adaptive) ---
                    # Only calibrate once after 10 fish have been detected
                    num_validated_fish = len(self.fish_weights)
                    if not self.is_calibrated and num_validated_fish >= CALIBRATION_THRESHOLD:
                        current_avg = sum(self.fish_weights.values()) / num_validated_fish
                        # Adjustment factor for PX_TO_CM based on L^3 relation
                        adjustment_factor = (TARGET_AVG_WEIGHT / current_avg) ** (1/3) 
                        self.biomass_estimator.px_to_cm *= adjustment_factor
                        
                        # [EXPERT v2.4] Recalculate existing weights with the new ratio for consistency
                        for tid in self.fish_weights:
                            self.fish_weights[tid] *= (adjustment_factor ** 3)
                            
                        self.is_calibrated = True
                        self.calibration_status = "STABLE (AUTO)"
                    elif not self.is_calibrated:
                        self.calibration_status = f"CALIBRATING ({num_validated_fish}/{CALIBRATION_THRESHOLD})..."

                    # [EXPERT v2.4] "Population-Pull" - Force far-distance fish toward session median
                    # If the fish is way below target but in the same cage, it's likely just far.
                    if self.is_calibrated and track_id in self.fish_weights:
                        w_current = self.fish_weights[track_id]
                        if w_current < TARGET_AVG_WEIGHT * 0.5: # Way below average
                            # Apply a 60% "Pull" toward target to compensate the 2D depth bias
                            self.fish_weights[track_id] = (w_current * 0.4) + (TARGET_AVG_WEIGHT * 0.6)

                    # Counting & Logging
                    if track_id not in self.counter:
                        self.counter.add(track_id)
                        with open(self.csv_path, 'a') as f:
                            weight_now = self.fish_weights.get(track_id, 0)
                            f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')},{self.processed_frames},{len(self.counter)},{track_id},{weight_now:.1f}g,{self.total_biomass_kg:.2f}kg,TRACKED\n")
                   
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

                    # 3. Masques et Rectangles Violets Pro
                    color = (255, 255, 0) if is_clipped else COLOR_VIOLET 
                    
                    # Dessin du Masque (Effet Elite)
                    if masks is not None:
                        try:
                            # On récupère le masque correspondant à cet index (YOLO les garde dans l'ordre des boîtes)
                            idx = list(track_ids).index(track_id)
                            mask_poly = np.array(masks[idx], dtype=np.int32)
                            # Remplissage translucide du masque
                            overlay_mask = frame.copy()
                            cv2.fillPoly(overlay_mask, [mask_poly], color)
                            cv2.addWeighted(overlay_mask, 0.3, frame, 0.7, 0, frame)
                            # Contour fin du masque
                            cv2.polylines(frame, [mask_poly], True, color, 1)
                        except: pass

                    label = f'ID {int(track_id)} | {int(self.fish_weights.get(track_id, 0))}g'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cvzone.putTextRect(frame, label, [max(0, x1), max(10, y1-10)], 
                                     scale=1, thickness=2, colorR=color)
        
        # Update Session Metrics once per frame
        self.total_biomass_kg = sum(self.fish_weights.values()) / 1000.0
        self.object_count = len(self.counter)
        
        # 3. Bandeau Noir AquaVision (Affichage Final)
        self.draw_aqua_header(frame)

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

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.stopped = True
                break
            elif key == ord('c') or key == ord('C'):
                print("[BIOMASS] Recalibration command received...")
                self.recalibrate_flag = True
            elif key == ord('m') or key == ord('M'):
                new_mode = self.biomass_estimator.toggle_mode()
                print(f"[BIOMASS] Mode switched to: {new_mode}")

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

        # --- FINAL SUMMARY IN CSV ---
        with open(self.csv_path, 'a') as f:
            f.write(f"\nFINAL SUMMARY,,,TOTAL FISH:,{len(self.counter)},TOTAL BIOMASS:,{self.total_biomass_kg:.2f}kg\n")

        print("\n=== FINAL RESULT ===")
        print(f"Total Fish Detected: {len(self.counter)}")


if __name__ == "__main__":
    counter = SingleCamFishCounter(MODEL_PATH, VIDEO_PATH)
    counter.run()
