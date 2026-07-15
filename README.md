# AquaVision Sea-Bream AI

AquaVision is a computer vision system for aquaculture monitoring focused on Sea Bream (`Sparus aurata`). The public release provides a complete local pipeline for fish detection, instance segmentation, multi-object tracking, counting, biomass estimation, and dashboard-based supervision from underwater video.

[![Status](https://img.shields.io/badge/Release-Public_2026-brightgreen)](https://github.com/Dourayed-Smari/Aquavision-seabream-AI)
[![Model](https://img.shields.io/badge/Model-YOLO11m--seg-blue)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack-blue)](https://github.com/ifzhang/ByteTrack)

## Pipeline overview

The current release is built around four main components:

1. `YOLO11m-seg` for fish detection and instance segmentation in underwater scenes.
2. `ByteTrack` for temporal identity tracking across video frames.
3. A biomass estimation module that combines geometric measurements, calibration logic, and temporal smoothing.
4. A Flask dashboard for video supervision and live indicator display.

## Repository contents

This public repository already includes:

- the pretrained segmentation model at `weights/bestmodel1.pt`
- three demonstration videos in `results/`
  - `AquaVision_Report_20260517_195747.mp4`
  - `AquaVision_Report_20260518_181423.mp4`
  - `AquaVision_Report_20260519_115305.mp4`

The main application entry points are:

- `core/Mainfishcount.py` for the local processing pipeline
- `dashboard/app.py` for the Flask supervision interface

## Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/Dourayed-Smari/Aquavision-seabream-AI.git
cd Aquavision-seabream-AI
pip install -r requirements.txt
```

For the dashboard session secret, define an environment variable before launch:

```bash
set FLASK_SECRET_KEY=your_secret_key_here
```

## Run

Run the local pipeline:

```bash
python core/Mainfishcount.py
```

Run the Flask dashboard:

```bash
python dashboard/app.py
```

## Technical highlights

- Underwater fish segmentation using `YOLO11m-seg`
- Identity-preserving tracking with `ByteTrack`
- Biomass estimation based on calibrated morphometric features
- Local execution oriented toward practical aquaculture supervision workflows

## Public release scope

This release contains the production-oriented codebase and selected demonstration assets only. Private engineering documents, LaTeX report sources, internal notes, and experimental development copies are excluded from publication.

## License

See `LICENSE` for repository licensing information.
