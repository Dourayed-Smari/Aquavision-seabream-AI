# AquaVision SeaBream Manager: AI-Driven Aquaculture Analysis

## Overview
This project is an advanced automated system for **Sea-Bream** (Dorade) detection, counting, and biomass estimation. It leverages state-of-the-art computer vision to provide aquaculture managers with precise real-time analytics from video inputs.

### Key Features
- **Ultra-High Sensitivity Detection**: Powered by a custom Kaggle-trained **YOLO Medium Model (`best+.pt`)** optimized for dense and distant Sea-Bream detection (1024px, 0.01 threshold).
- **Masterclass Cinematic Tracking**: Uses heavily customized **ByteTrack** with an extended memory buffer (8s survival) and a non-overlapping "Sniper" UI (centroids & short tracking tails) to cleanly visualize tracking without clutter.
- **CPU Anti-Lag & Slow-Mo Export**: Multi-threaded processing architecture ensures real-time stability, while silently exporting an automated **Slow-Motion MP4 (0.25x)** video and timestamped **CSV Data Logs** to the `/results` directory.
- **Biomass Measurement (Upcoming)**: Architecture ready for pixel-to-gram allometric regression based on dynamic bounding box geometry extraction.

## Project Architecture
The project is structured to separate the core detection logic, the future biomass calculation engine, and the web interface.

```text
AquaVision/
├── core/                 # Core logic and settings
│   ├── Mainfishcount.py  # Multi-threaded master application
│   └── custom_bytetrack.yaml # Heavy-duty tracking configurations
├── web/                  # Web dashboard (Future implementation)
│   ├── app.py            # Main web entry point
│   ├── static/           # CSS/JS assets
│   └── templates/        # HTML templates
├── weights/              # Training weights (.pt files)
├── data/                 # Sample video datasets
├── results/              # Directory for processed output videos
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
├── LICENSE               # GNU GPL v3.0 License
└── README.md             # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Dourayed-Smari/Aquavision-seabream-AI.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```bash
python core/Mainfishcount.py
```

## Note on Model Weights
Due to GitHub's file size limits, the pre-trained weights are not included in this repository. 
> [!IMPORTANT]
> Please download the custom expert weights (`best+.pt`) from [https://drive.google.com/file/d/1BokebUAWlyLInyMWO0Htdl-LMFQwedU9/view?usp=drive_link] and place them in the `/weights` directory.

## Authors
- **Dourayed Smari** - *Lead Developer & AI Training*

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
