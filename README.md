# AquaVision : Sea-Bream AI Monitoring System 🐟⚓

AquaVision is an expert-grade aquaculture monitoring platform focused on the precision counting, tracking, and biomass estimation of **Sea Bream** (*Sparus aurata*). This project represents a state-of-the-art implementation of computer vision applied to industrial aquaculture.

[![Status](https://img.shields.io/badge/Version-2.4.2_Adaptive-brightgreen)](https://github.com/Dourayed-Smari/Aquavision-seabream-AI)
[![Framework](https://img.shields.io/badge/AI-YOLOv8_|_OpenVino-blueviolet)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack_|_SU--T-blue)](https://github.com/ifzhang/ByteTrack)

---

## 🌊 Core Components of the Pipeline

The system utilizes a multi-stage pipeline designed for robustness in challenging underwater environments:

### 1. Object Detection (YOLOv8)
Powered by a custom-trained **YOLOv8m** model optimized for the specific morphology of Sea Bream. The weights in `best+.pt` were trained on **Kaggle** for over **11 hours** using a high-resolution input of **`imgsz=1024`**. This extensive training ensures high precision for detecting small features and overlapping individuals even in low-visibility underwater environments.

### 2. Multi-Object Tracking (MOT)
The current stable release is strictly powered by **ByteTrack**. Other architectures were tested but are currently **inactive** and preserved for research purposes only:
*   **ByteTrack (STABLE | DEFAULT)** : High performance and identity stability in dense schools.
*   **BoT-SORT (Testing Phase)** : Observed improved identity but higher latency. Currently inactive.
*   **SU-T (Experimental Research)** : Scale-aware Unscented Tracker using UKF filters. Specialized research module, currently inactive in the main branch (see `core/trackers/`).

### 3. Deep-Z Biomass Estimation (v2.4)
Our proprietary biomass module solves the 2D depth ambiguity through:
*   **Population-Pull Logic** : Normalizes individual weight estimates toward the session's lot median, ensuring biological consistency across the entire cage.
*   **Hyperbolic Perspective Correction** : A 40% depth compensation factor ($K_{depth}$) that restores the true weight of far-away fish.
*   **Adaptive Auto-Calibration** : Dynamic $PX \to CM$ ratio adjustment triggered after 10 validated detections.

---

## 🚀 Getting Started

### 📦 Installation
1.  **Clone the Repository** :
    ```bash
    git clone https://github.com/Dourayed-Smari/Aquavision-seabream-AI.git
    ```
2.  **Install Requirements** :
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download AI Weights** : 
    Download the trained model `best+.pt` and place it in the `weights/` directory.
    🔗 **[DOWNLOAD MODEL (Google Drive)](https://drive.google.com/file/d/1BokebUAWlyLInyMWO0Htdl-LMFQwedU9/view?usp=drive_link)**

### 📈 Running the System
```bash
python core/Mainfishcount.py
```

---

## 📑 Key Features
*   **Adaptive UI** : Real-time status indicators (CALIBRATING vs STABLE) and high-visibility Cyan safety alerts.
*   **Biological Safety Valve** : Automatic clipping of outliers (100g - 1200g) to prevent perspective-induced errors.
*   **Cinematic Reporting** : Automatic export of results to `results/` with CSV logs and MP4 summaries including **Extreme Slow-Mo Ramping** on detection events.

---

## ⚖️ Acknowledgments & Legal
This project stands on the shoulders of giants. We wish to dedicate a special credit to the **SU-T (Scale-aware Unscented Tracker)** architecture. Our experimental phase deeply utilized the UKF-based orientation and scale estimation logic for high-precision underwater tracking.

All tracking modules and biological constraints have been tailored for the specific needs of Mediterranean Sea Bream aquaculture.

---
*Maintained by Dourayed Smari | AquaVision AI Team 2026*
