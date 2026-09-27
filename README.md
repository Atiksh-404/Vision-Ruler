# VisionRuler

VisionRuler is a real-time object measurement app built with Python, OpenCV, and NumPy. It uses a webcam to detect and track an object's outline, then estimates its dimensions.

## Features

- Live webcam measurement
- Reference-card calibration for estimated centimeter measurements
- Rotated object measurement
- Object tracking and a live status dashboard
- Hand-skin suppression to help separate held objects

## Requirements

- Python
- A webcam
- A standard bank card for calibration

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Controls

- `C` - Calibrate with the reference card
- `R` - Reset calibration
- `D` - Save a detection debug image
- `Q` - Quit

## Notes

Measurements are estimates. Lighting, glare, covered edges, and the card and object being at different distances from the camera can affect accuracy.