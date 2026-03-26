# AquaVision SeaBream Manager: AI-Driven Aquaculture Analysis

## Overview
This project is an advanced automated system for **Sea-Bream** (Dorade) detection, counting, and biomass estimation. It leverages state-of-the-art computer vision to provide aquaculture managers with precise real-time analytics from video inputs.

### Key Features
- **Precision Detection**: Powered by an **Advanced Sea-Bream YOLO Model** for accurate identification.
- **Dynamic Tracking**: Implements the **SORT** (Simple Online and Realtime Tracking) algorithm to monitor individual fish movements and prevent count duplication. 
- **Biomass Measurement**: Real-time estimation of length (cm) and weight (gm) based on calibrated anatomical ratios.
- **Analytics Dashboard**: Instant analytics (Count, Total Biomass, Average Size) and professional result videos (stored in the `/results` directory).

## Project Architecture
The project is structured to separate the core detection logic, the future biomass calculation engine, and the web interface.

```text
AquaVision/
├── Mainfishcount.py      # Multi-threaded real-time entry point
├── core/                 # Core processing modules
│   ├── sort.py           # SORT tracking implementation
│   └── biomass.py        # Future biomass calculation module
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
python Mainfishcount.py
```

## Note on Model Weights
Due to GitHub's file size limits, the pre-trained weights are not included in this repository. 
> [!IMPORTANT]
> Please download the weights (`best@.pt`) from [https://drive.google.com/file/d/1pJOQwtCTu1EqEeNMrZNgPDasq22wAbMb/view] and place them in the `/weights` directory.

## Authors
- **Dourayed Smari** - *Lead Developer & AI Training*

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
