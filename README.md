# AquaVision : Sea-Bream AI Monitoring 🐟⚓

AquaVision is an advanced aquaculture monitoring system designed for the precision detection, tracking, and biomass estimation of Sea Bream (*Sparus aurata*).

![Banner](https://img.shields.io/badge/Status-Version_2.4_Stable-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-blueviolet)

## 🚀 Key Features

*   **Deep-Z Biomass Estimation (v2.4)** : Non-linear perspective correction and population-based weight synchronization for high precision even with 2D cameras.
*   **Adaptive Auto-Calibration** : The system automatically learns the cage scale after 10 fish detections, ensuring consistency across different video sources.
*   **High-Precision Tracking** : Powered by ByteTrack with custom biological filters (Pose-Score) to ensure only valid profiles are measured.
*   **Time-Ramp Reporting** : Professional video exports with cinematic slow-motion during high-activity detections.
*   **Interactive Calibration** : On-the-fly pixel-to-cm adjustment using reference objects (Hotkey 'C').

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Dourayed-Smari/Aquavision-seabream-AI.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your model in `weights/best+.pt`.

## 📈 Usage

Run the main counter script:
```bash
python core/Mainfishcount.py
```

## 🔬 Scientific Methodology

The system uses the allometric growth formula $W = a \cdot L^b$ with $a=0.012$ and $b=3.0$ for Sea Bream. It incorporates a **Population-Pull** force that normalizes individual measurements toward the lot median, overcoming 2D depth ambiguity.

---
*Developed for the AquaVision SeaBream AI Project.*
