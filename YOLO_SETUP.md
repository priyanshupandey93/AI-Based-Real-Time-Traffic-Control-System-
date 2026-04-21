# VC_1.py - YOLO Vehicle Counter Setup Guide

## Overview
VC_1.py now uses **YOLOv8 (You Only Look Once)** for accurate real-time vehicle detection, replacing the background subtraction method. YOLO is a state-of-the-art deep learning model that achieves much higher accuracy and robustness.

## Installation Requirements

### Step 1: Install Required Packages
Run this command in your terminal:

```bash
pip install ultralytics torch torchvision opencv-python numpy
```

Or use the requirements file:
```bash
pip install -r REQUIREMENTS.txt
```

### Step 2: First Run
- The first time you run VC_1.py, it will automatically download the YOLOv8 Nano model (~40 MB)
- This is a one-time download that will be cached locally
- Subsequent runs will be faster

## Running VC_1.py

```bash
python VC_1.py
```

## Features

✓ **YOLOv8 Nano Model** - Fast inference on CPU or GPU
✓ **Vehicle Detection** - Detects cars, motorcycles, buses, trucks
✓ **Accurate Tracking** - Tracks vehicles across frames
✓ **Single Road Counting** - Counts vehicles crossing a detection line
✓ **Real-time Display** - Shows detections with bounding boxes

## Controls

- **ESC** - Exit the program
- **'+'** - Slow down playback
- **'-'** - Speed up playback
- **'0'** - Reset to real-time speed

## Output

The program displays:
- **Green boxes** - Currently tracking vehicles
- **Blue boxes** - Vehicles already counted
- **Red flash** - Newly detected vehicle
- **Orange line** - Detection threshold line
- **Total count** - Number of vehicles detected at bottom

## Performance Notes

- **CPU Mode** - Takes 2-5 seconds per frame
- **GPU Mode** - Much faster if NVIDIA GPU with CUDA is available
- For real-time performance, use GPU or consider YOLOv8n (nano) model

## Troubleshooting

**Issue: ModuleNotFoundError: No module named 'ultralytics'**
- Solution: Install with `pip install ultralytics`

**Issue: Model downloading fails**
- Solution: Check internet connection, try again later
- The model downloads to ~/.yolo/weights/

**Issue: Slow performance**
- Solution: If using CPU, this is normal. Consider using YOLOv8n (nano)
- If GPU available, ensure CUDA is installed

## Parameters You Can Adjust

In VC_1.py:
- `CONFIDENCE_THRESHOLD = 0.45` - Lower = more detections (0.0-1.0)
- `VEHICLE_CLASSES = [2, 3, 5, 7]` - Classes to detect (COCO IDs)
- `play_speed = 3` - Initial playback speed
- `count_line_position = int(fh * 0.6)` - Position of detection line

