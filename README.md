# 🚦 AI-Based Real-Time Traffic Control System

An intelligent, adaptive traffic signal management system that uses **YOLOv8 deep learning** to count vehicles across 4 intersecting roads in real-time and dynamically allocates green-light durations based on live traffic density.

---

## 📌 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [File Descriptions](#file-descriptions)
- [Controls & Keyboard Shortcuts](#controls--keyboard-shortcuts)
- [Configuration & Tunable Parameters](#configuration--tunable-parameters)
- [Troubleshooting](#troubleshooting)

---

## 📖 Overview

This project implements a **4-lane AI-powered traffic control system** for a standard road intersection. Instead of fixed-timer signals, the system:

1. Continuously counts vehicles from **4 live video feeds** (one per road direction) using **YOLOv8 object detection**.
2. Feeds real-time vehicle counts into a **Simulation Dashboard** that dynamically calculates green-light duration proportional to traffic density.
3. Cycles through the 4 lanes (North → South → West → East), giving each lane a green phase weighted by its vehicle count.
4. After each green phase ends, that lane's count is **reset to zero**, so the next cycle accurately measures fresh arrivals.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│         (Single entry-point — launches both windows)    │
└────────────────┬───────────────────────┬────────────────┘
                 │                       │
    ┌────────────▼──────────┐  ┌─────────▼──────────────┐
    │   main_dashboard.py   │  │ simulation_dashboard.py │
    │  4-Camera Detection   │  │  Adaptive Signal Timer  │
    │  Feed (Writer)        │  │  (Reader / Controller)  │
    └────────────┬──────────┘  └─────────┬──────────────┘
                 │                       │
         ┌───────▼───────────────────────▼──────┐
         │           count_store.py              │
         │  Shared inter-process data bridge     │
         │  (.traffic_counts.json  — atomic I/O) │
         └───────────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────┐
    │  VC_1.py / VC_2.py / VC_3.py / VC_4.py  │
    │  Individual YOLOv8 vehicle counters       │
    │  (one per camera / road direction)        │
    └──────────────────────────────────┘
```

---

## 📁 Project Structure

```
Main_File/
│
├── main.py                  # ▶ Entry point — launches both dashboards
├── main_dashboard.py        # 4-camera live video monitoring dashboard
├── simulation_dashboard.py  # Adaptive traffic signal simulation dashboard
├── count_store.py           # Shared inter-process vehicle count bridge
│
├── VC_1.py                  # Vehicle counter — Camera 1 (North)
├── VC_2.py                  # Vehicle counter — Camera 2 (South)
├── VC_3.py                  # Vehicle counter — Camera 3 (West)
├── VC_4.py                  # Vehicle counter — Camera 4 (East)
│
├── yolov8n.pt               # YOLOv8 Nano pre-trained model weights
│
├── test_videos/             # Sample traffic video files
│   ├── 1.mp4                # Test video — Camera 1 (North)
│   ├── 2.mp4                # Test video — Camera 2 (South)
│   ├── 3.mp4                # Test video — Camera 3 (West)
│   └── 4.mp4                # Test video — Camera 4 (East)
│
├── .traffic_counts.json     # Auto-generated: live vehicle counts (shared)
├── .reset_signals.json      # Auto-generated: per-lane reset flags (shared)
│
├── REQUIREMENTS.txt         # Python package dependencies
└── YOLO_SETUP.md            # YOLO model setup and usage guide
```

---

## ⚙️ Requirements

### 🐍 Programming Language

| Requirement | Version         |
|-------------|-----------------|
| Python      | **3.8 or later** (3.10+ recommended) |

### 📦 Python Libraries

| Library          | Version     | Purpose                                      |
|------------------|-------------|----------------------------------------------|
| `opencv-python`  | ≥ 4.5.0     | Video capture, image processing, GUI display |
| `numpy`          | ≥ 1.19.0    | Numerical computation for detection logic    |
| `ultralytics`    | ≥ 8.0.0     | YOLOv8 model (vehicle detection & tracking)  |
| `torch`          | ≥ 1.9.0     | Deep learning backend for YOLO               |
| `torchvision`    | ≥ 0.10.0    | Vision transforms required by PyTorch        |
| `tkinter`        | Built-in     | GUI framework for dashboards (included with Python) |
| `threading`      | Built-in     | Multi-threaded video stream processing       |
| `subprocess`     | Built-in     | Launching child processes from `main.py`     |
| `json`           | Built-in     | Shared inter-process data store              |

### 💻 Hardware

| Component | Minimum              | Recommended                    |
|-----------|----------------------|--------------------------------|
| CPU       | Any modern CPU       | Multi-core (4+ cores)          |
| RAM       | 4 GB                 | 8 GB+                          |
| GPU       | *(Optional)*         | NVIDIA GPU with CUDA (for fast inference) |
| Storage   | 500 MB free          | For model weights + video files |

> **Note:** The system runs on CPU by default. A CUDA-enabled NVIDIA GPU significantly improves performance (real-time vs. 2–5 sec/frame).

---

## 🔧 Installation

### Step 1: Clone or Download the Project

```bash
# Clone via Git
git clone https://github.com/Abhay0222/AI-Based-Real-Time-Traffic-Control-System-.git
cd AI-Based-Real-Time-Traffic-Control-System-
```

Or simply **download and extract** the ZIP from GitHub.

---

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it — Windows
venv\Scripts\activate

# Activate it — macOS / Linux
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r REQUIREMENTS.txt
```

Or install manually:

```bash
pip install opencv-python>=4.5.0 numpy>=1.19.0 ultralytics>=8.0.0 torch>=1.9.0 torchvision>=0.10.0
```

---

### Step 4: GPU Support (Optional but Recommended)

If you have an **NVIDIA GPU**, install the CUDA-enabled version of PyTorch for much faster inference:

```bash
# Visit https://pytorch.org/get-started/locally/ to get the right command for your CUDA version.
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 5: YOLOv8 Model Weights

The file `yolov8n.pt` (YOLOv8 Nano model, ~6 MB) is already included in the project directory.

If it is missing, it will be **automatically downloaded** on first run from the Ultralytics servers (requires internet connection).

---

### Step 6: Add Test Videos

Place your traffic video files inside the `test_videos/` folder:

| File Name | Camera          | Direction |
|-----------|-----------------|-----------|
| `1.mp4`   | Camera 1        | North     |
| `2.mp4`   | Camera 2        | South     |
| `3.mp4`   | Camera 3        | West      |
| `4.mp4`   | Camera 4        | East      |

> The videos loop automatically, so any traffic footage works.

---

## ▶️ How to Run

### ✅ Recommended: Run Everything with One Command

```bash
python main.py
```

This single command:
1. Resets all vehicle counts to zero.
2. Opens the **Vehicle Detection Dashboard** (4-camera live feed with bounding boxes).
3. Opens the **Signal Simulation Dashboard** (live counts table + adaptive signal timer).

---

### 🔬 Run Individual Components (Optional)

| Script                      | Command                           | Description                        |
|-----------------------------|-----------------------------------|------------------------------------|
| Vehicle Detection Dashboard | `python main_dashboard.py`        | Watch all 4 camera feeds live      |
| Simulation Dashboard only   | `python simulation_dashboard.py`  | Signal timer (reads saved counts)  |
| Camera 1 counter only       | `python VC_1.py`                  | Standalone North road counter      |
| Camera 2 counter only       | `python VC_2.py`                  | Standalone South road counter      |
| Camera 3 counter only       | `python VC_3.py`                  | Standalone West road counter       |
| Camera 4 counter only       | `python VC_4.py`                  | Standalone East road counter       |

---

## ⚡ How It Works

### 1. Vehicle Detection (YOLOv8)
Each `VC_*.py` script processes a video stream frame-by-frame using **YOLOv8 Nano** model. It detects:

- 🚗 Cars &nbsp;&nbsp; (COCO class 2)
- 🏍️ Motorcycles &nbsp;&nbsp; (COCO class 3)
- 🚌 Buses &nbsp;&nbsp; (COCO class 5)
- 🚛 Trucks &nbsp;&nbsp; (COCO class 7)

A **virtual detection line** is drawn across each frame. When a vehicle's centroid crosses this line, it is counted once and its ID is added to the "counted" set to prevent double-counting.

### 2. Multi-Camera Dashboard (`main_dashboard.py`)
A `tkinter`-based GUI that runs all 4 video streams simultaneously in separate threads (`VideoStreamThread`). Each thread:
- Reads frames using OpenCV.
- Runs YOLO inference on each frame.
- Pushes the updated count to `count_store.py` via `update_count()`.

### 3. Shared Data Bridge (`count_store.py`)
Acts as the **inter-process communication layer** between the two dashboards:
- Uses **atomic file writes** (write to temp → rename) to prevent data corruption.
- The simulation dashboard **polls** it every 200 ms.
- When a green phase ends, `request_reset(direction)` zeroes that lane's count and sets a reset flag, causing the video thread to restart its baseline count.

### 4. Adaptive Signal Simulation (`simulation_dashboard.py`)
- Reads live counts from `count_store.py`.
- Calculates each lane's **green time** proportionally:
  ```
  Green Time (lane) = (Lane Count / Total Count) × Total Cycle Time
  ```
- Applies **minimum (10 s) and maximum (60 s)** green time bounds.
- Cycles through lanes: North → South → West → East → repeat.
- Displays a live countdown and color-coded signal status (🟢 GREEN / 🔴 RED).

---

## 🗂️ File Descriptions

| File | Role |
|------|------|
| `main.py` | Master entry point. Resets counts, launches both dashboards as subprocesses |
| `main_dashboard.py` | Tkinter GUI: 4 video feeds with real-time detection overlays; writes to count_store |
| `simulation_dashboard.py` | Tkinter GUI: adaptive signal timer that reads counts and runs the signal cycle |
| `count_store.py` | Thread/process-safe JSON-backed data bridge (atomic writes + retry reads) |
| `VC_1.py` | Standalone YOLOv8 vehicle counter for Camera 1 (North). Uses `test_videos/1.mp4` |
| `VC_2.py` | Standalone YOLOv8 vehicle counter for Camera 2 (South). Uses `test_videos/2.mp4` |
| `VC_3.py` | Standalone YOLOv8 vehicle counter for Camera 3 (West). Uses `test_videos/3.mp4` |
| `VC_4.py` | Standalone YOLOv8 vehicle counter for Camera 4 (East). Uses `test_videos/4.mp4` |
| `yolov8n.pt` | Pre-trained YOLOv8 Nano model weights (~6 MB). Auto-downloaded if missing |
| `.traffic_counts.json` | Auto-generated runtime file. Stores `{North, South, West, East}` counts |
| `.reset_signals.json` | Auto-generated runtime file. Stores per-lane reset flags |

---

## ⌨️ Controls & Keyboard Shortcuts

### In Individual `VC_*.py` Windows

| Key | Action |
|-----|--------|
| `ESC` | Exit / close the window |
| `+` or `=` | Slow down video playback |
| `-` | Speed up video playback |
| `0` | Reset to real-time playback speed |

### In `simulation_dashboard.py`

| Button | Action |
|--------|--------|
| ▶ Start Simulation | Begin the adaptive signal cycle |
| ⏹ Stop Simulation | Pause / stop the signal cycle |

---

## 🛠️ Configuration & Tunable Parameters

### In `VC_*.py` files

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.40` | YOLO detection confidence (0.0 – 1.0). Lower = more detections |
| `VEHICLE_CLASSES` | `[2, 3, 5, 7]` | COCO class IDs to detect (car, motorcycle, bus, truck) |
| `play_speed` | `1.5` | Video playback multiplier (1 = real-time, >1 = slower frame rate) |
| `count_line_position` | `int(fh * 0.65)` | Y-position of the counting line as fraction of frame height |
| `MOTION_THRESHOLD` | `5` | Min pixel movement to classify a vehicle as "moving" |
| `MAX_MISSING_FRAMES` | `15` | Frames a tracked vehicle can be absent before it is dropped |

### In `simulation_dashboard.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| Minimum green time | `10 s` | No lane gets less than 10 seconds |
| Maximum green time | `60 s` | No lane gets more than 60 seconds |
| Poll interval | `200 ms` | How often the simulation reads fresh counts |

---

## 🔍 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### ❌ `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### ❌ YOLOv8 model download fails
- Check your internet connection.
- Ensure `yolov8n.pt` exists in the project root directory.
- Download manually from: https://github.com/ultralytics/assets/releases

### ⚠️ Very slow performance (2–5 sec per frame)
- This is normal for **CPU-only** machines.
- Install a **CUDA-enabled GPU** version of PyTorch for real-time performance.
- Alternatively, use a lower-resolution video input.

### ⚠️ Video not found error
- Ensure your video files (`1.mp4`, `2.mp4`, `3.mp4`, `4.mp4`) are in the `test_videos/` subdirectory.
- Check the file path in each `VC_*.py`: `cv2.VideoCapture('test_videos/X.mp4')`.

### ⚠️ Dashboard windows don't appear
- Make sure `tkinter` is installed. On some Linux systems: `sudo apt-get install python3-tk`
- On Windows, `tkinter` is bundled with the standard Python installer.

### ⚠️ Counts not updating in Simulation Dashboard
- Make sure `main_dashboard.py` is running (it writes the counts).
- Both scripts must be launched together using `python main.py`.

---

## 👤 Authors

**Abhay Singh**
[LinkedIn](https://www.linkedin.com/in/abhay-singh-73921827a/)
**Priyanshu Pandey**
[LinkedIn](https://www.linkedin.com/in/priyanshu-pandey-93gn/)
**Harsh Kumar Maddheshiya**
[LinkedIn](https://www.linkedin.com/in/harsh-kumar-maddheshiya-0479a9253/)
---

## 📄 License

This project is developed for academic purposes as part of a Major Project submission.

---

*Last updated: April 2026*
