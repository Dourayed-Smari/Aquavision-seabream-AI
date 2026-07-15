import os
import sys
import cv2
import threading
import queue
import time
from flask import Flask, render_template, Response, jsonify, redirect, request, session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.Mainfishcount import SingleCamFishCounter

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "CHANGE_ME_IN_PROD")

MODEL_PATH  = "weights/bestmodel1.pt"
UPLOAD_DIR  = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Global Engine (on-demand) ────────────────────────────────────────
counter_engine: 'WebFishCounter | None' = None
engine_lock = threading.Lock()


class WebFishCounter(SingleCamFishCounter):
    def __init__(self, model_path, video_source):
        super().__init__(model_path, video_source)

        # 🧠 Force ML Regression Engine
        self.biomass_estimator.current_mode = self.biomass_estimator.MODE_ML
        print(f"[ENGINE] Moteur Biomasse : {self.biomass_estimator.current_mode}")

        self.web_frame        = None
        self.metrics          = {"fish_count": 0, "avg_weight": 0.0, "total_biomass": 0.0}
        self.loop_cache       = []
        self.first_loop_done  = False
        self.final_metrics    = None
        self.cache_idx        = 0
        self.final_snapshot_jpg = None

        # 📊 Telemetry
        self.start_time         = time.time()
        self.total_frames_proc  = 0
        self.current_latency    = 0
        self._was_calibrated    = False
        self.calibration_frame  = None

    # ── Override run(): no OpenCV window in web context ──────────────
    def run(self):
        threads = [
            threading.Thread(target=self.capture_thread, daemon=True),
            threading.Thread(target=self.process_thread,  daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.stopped = True
        self.writer.release()

    # ── Overridden capture thread (loop-aware) ────────────────────────
    def capture_thread(self):
        delay = 1.0 / self.fps
        while not self.stopped:
            if self.first_loop_done:
                time.sleep(1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                print("\n[ENGINE] Analyse complète — snapshot certifié sauvegardé.")
                self.frame_queue.join()
                self.final_metrics = self.metrics.copy()
                self.first_loop_done = True
                continue
            time.sleep(delay)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.task_done()
                except: pass
            try: self.frame_queue.put(frame)
            except: pass
        self.cap.release()

    # ── Process thread ────────────────────────────────────────────────
    def process_thread(self):
        while not self.stopped:
            if self.first_loop_done:
                break
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.time()
            annotated = self.process_frame(frame.copy())
            self.current_latency = round((time.time() - t0) * 1000)

            resized = cv2.resize(annotated, (1280, 720))
            _, buf  = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.loop_cache.append(buf)
            self.final_snapshot_jpg = buf

            avg_w = (sum(self.fish_weights.values()) / len(self.fish_weights)) if self.fish_weights else 0
            self.total_frames_proc += 1

            if self.is_calibrated and not self._was_calibrated:
                self.calibration_frame = self.total_frames_proc
                self._was_calibrated   = True

            self.metrics = {
                "fish_count":    len(self.counter),
                "avg_weight":    round(avg_w, 1),
                "total_biomass": round(self.total_biomass_kg, 2),
            }
            with self.lock:
                self.web_frame = resized
            self.processed_frames += 1
            self.frame_queue.task_done()

    # ── Telemetry ─────────────────────────────────────────────────────
    def get_telemetry(self):
        uptime = int(time.time() - self.start_time)
        h, rem = divmod(uptime, 3600)
        m, s   = divmod(rem, 60)
        return {
            "frame_count":       self.total_frames_proc,
            "latency_ms":        self.current_latency,
            "loop_status":       "REPLAY_LOOP" if self.first_loop_done else "ANALYSIS",
            "engine":            self.biomass_estimator.current_mode.replace("-", "_"),
            "uptime":            f"{h:02d}:{m:02d}:{s:02d}",
            "calibration_frame": self.calibration_frame,
            "ready":             self.web_frame is not None,
        }


# ── Frame Generator ───────────────────────────────────────────────────
def generate_frames():
    global counter_engine
    while True:
        if counter_engine is None:
            time.sleep(0.2)
            continue
        if counter_engine.first_loop_done and counter_engine.loop_cache:
            buf   = counter_engine.loop_cache[counter_engine.cache_idx]
            frame = buf.tobytes()
            counter_engine.cache_idx = (counter_engine.cache_idx + 1) % len(counter_engine.loop_cache)
            time.sleep(1.0 / counter_engine.fps)
        else:
            with counter_engine.lock:
                if counter_engine.web_frame is None:
                    time.sleep(0.1)
                    continue
                _, buffer = cv2.imencode('.jpg', counter_engine.web_frame)
                frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/upload', methods=['POST'])
def upload():
    global counter_engine

    if 'video' not in request.files:
        return redirect('/')

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect('/')

    # Save uploaded video
    save_path = os.path.join(UPLOAD_DIR, 'session_input.mp4')
    video_file.save(save_path)
    print(f"[UPLOAD] Vidéo sauvegardée → {save_path}")

    # Instantiate and start engine
    with engine_lock:
        counter_engine = WebFishCounter(MODEL_PATH, save_path)
        engine_thread  = threading.Thread(target=counter_engine.run, daemon=True)
        engine_thread.start()
        print("[ENGINE] Moteur IA démarré sur la vidéo uploadée.")

    return redirect('/dashboard')


@app.route('/dashboard')
def dashboard():
    global counter_engine
    if counter_engine is None:
        return redirect('/')
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_metrics')
def get_metrics():
    global counter_engine
    if counter_engine is None:
        return jsonify({"fish_count": 0, "avg_weight": 0, "total_biomass": 0, "ready": False})
    base = counter_engine.final_metrics if counter_engine.first_loop_done else counter_engine.metrics
    return jsonify({**base, **counter_engine.get_telemetry()})


@app.route('/get_final_snapshot')
def get_final_snapshot():
    global counter_engine
    if counter_engine and counter_engine.final_snapshot_jpg is not None:
        return Response(counter_engine.final_snapshot_jpg.tobytes(), mimetype='image/jpeg')
    return "Non disponible", 404


@app.route('/reset')
def reset():
    global counter_engine
    with engine_lock:
        if counter_engine:
            counter_engine.stopped = True
        counter_engine = None
    return redirect('/')


if __name__ == '__main__':
    print("\n🚀  AQUAVISION | H2A Groupe · Galaxy Edition v11.0")
    print("    Accédez à : http://localhost:5001\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
