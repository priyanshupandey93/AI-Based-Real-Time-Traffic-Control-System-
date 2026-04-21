import cv2
import numpy as np

# Import YOLO for advanced vehicle detection
try:
    from ultralytics import YOLO
    print("Loading YOLOv8 Nano model for vehicle detection...")
    model = YOLO('yolov8n.pt')  # nano model for fast inference
    print("✓ YOLOv8 model loaded successfully!\n")
    YOLO_AVAILABLE = True
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics torch torchvision")
    YOLO_AVAILABLE = False
except Exception as e:
    print(f"ERROR loading YOLO model: {e}")
    YOLO_AVAILABLE = False

if not YOLO_AVAILABLE:
    print("YOLO model unavailable. Please install ultralytics.")
    exit()

# Video Capture
cap = cv2.VideoCapture('test_videos/1.mp4')

# YOLO detection parameters
CONFIDENCE_THRESHOLD = 0.40  # Increased threshold to ignore static sideboards and false positives
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO class IDs: car(2), motorcycle(3), bus(5), truck(7)

# Motion detection parameters
MOTION_THRESHOLD = 5  # Minimum pixels of centroid movement to be considered "moving"
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if metadata missing
delay = int(1000 / fps)                # ms per frame

cv2.namedWindow('VC_1 - Single Road Vehicle Counter', cv2.WINDOW_NORMAL)
cv2.resizeWindow('VC_1 - Single Road Vehicle Counter', 640, 360)
print("=" * 60)
print("SINGLE ROAD VEHICLE COUNTER - VC_1.py")
print("=" * 60)
print("Controls: '+' increase speed (slower), '-' decrease speed (faster), '0' reset to real-time\n")

# Counter and state
vehicle_count = 0
# Tracker variables for Vehicle IDs
tracked_vehicles = {}  # {track_id: (cx, cy, missing_frames)}
counted_ids = set()    # Keep track of counted vehicles to prevent double counting
next_vehicle_id = 1
MAX_MISSING_FRAMES = 15

# Playback speed multiplier: 1 = real-time, >1 = slower playback
play_speed = 1.5  # Playback speed

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    fh, fw = frame.shape[:2]
    if 'count_line_position' not in globals():
        count_line_position = int(fh * 0.65)  # Moved line slightly down, closer to camera

    # YOLO-based vehicle detection
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    frame_dets = []  # list of temporary detections for this frame
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Filter by vehicle class
            if class_id not in VEHICLE_CLASSES:
                continue
            
            # Calculate dimensions
            wbox = x2 - x1
            hbox = y2 - y1
            
            # Filter out detections that are too far away (too small)
            if wbox < 60 or hbox < 60:
                continue
            
            # Calculate centroid
            cx = int(x1 + wbox / 2)
            cy = int(y1 + hbox / 2)
            
            # Filter out sideboards on the extreme left/right edges
            if cx < int(fw * 0.15) or cx > int(fw * 0.85):
                continue
            
            frame_dets.append((x1, y1, wbox, hbox, cx, cy))

    current_detections = []  # list of (x,y,w,h,cx,cy, vid, last_cx, last_cy)
    new_tracked_vehicles = {}
    unassigned_dets = frame_dets.copy()
    
    # 1. Match existing tracked vehicles (sort by missing_frames so active get priority)
    for t_id, (tcx, tcy, missing) in sorted(tracked_vehicles.items(), key=lambda x: x[1][2]):
        if not unassigned_dets:
            if missing < MAX_MISSING_FRAMES:
                new_tracked_vehicles[t_id] = (tcx, tcy, missing + 1)
            continue
            
        min_dist = float('inf')
        best_match_idx = -1
        
        for i, (x1, y1, wbox, hbox, cx, cy) in enumerate(unassigned_dets):
            dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
            if dist < 100 and dist < min_dist:  # 100 pixels threshold for matching fast-moving vehicles near camera
                min_dist = dist
                best_match_idx = i
                
        if best_match_idx != -1:
            x1, y1, wbox, hbox, cx, cy = unassigned_dets.pop(best_match_idx)
            new_tracked_vehicles[t_id] = (cx, cy, 0)
            current_detections.append((x1, y1, wbox, hbox, cx, cy, t_id, tcx, tcy))
        elif missing < MAX_MISSING_FRAMES:
            new_tracked_vehicles[t_id] = (tcx, tcy, missing + 1)
            
    # 2. Assign new IDs to remaining detections
    for x1, y1, wbox, hbox, cx, cy in unassigned_dets:
        assigned_id = next_vehicle_id
        next_vehicle_id += 1
        new_tracked_vehicles[assigned_id] = (cx, cy, 0)
        current_detections.append((x1, y1, wbox, hbox, cx, cy, assigned_id, cx, cy))
        
    tracked_vehicles = new_tracked_vehicles

    # Check for vehicles crossing the detection line
    for x, y, wbox, hbox, cx, cy, vid, last_cx, last_cy in current_detections:
        # Check if crossed the detection line (vector crossing correctly captures fast cars jumping it)
        if (last_cy <= count_line_position and cy > count_line_position) or \
           (last_cy >= count_line_position and cy < count_line_position):
            if vid not in counted_ids:
                vehicle_count += 1
                counted_ids.add(vid)
                print(f"[Frame {frame_idx}] Vehicle VID {vid} crossed line | Total Vehicles: {vehicle_count}")

    # Draw vehicles on frame
    for x, y, wbox, hbox, cx, cy, vid, last_cx, last_cy in current_detections:
        # Check if moving based on its last known position
        dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
        is_moving = dist >= MOTION_THRESHOLD
        
        # Color: Green = moving, Yellow = stationary
        color = (0, 255, 0) if is_moving else (0, 255, 255)
        
        cv2.rectangle(frame, (x, y), (x + wbox, y + hbox), color, 2)
        
        # Display the Vehicle ID (VID)
        cv2.putText(frame, f"VID {vid}", (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # draw count line
    cv2.line(frame, (25, count_line_position), (fw - 25, count_line_position), (255, 127, 0), 3)
    cv2.putText(frame, "Detection Line", (30, count_line_position - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 127, 0), 1)

    # Show total count at bottom
    count_text = f"Total Vehicles Counted: {vehicle_count}"
    (tx, ty), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    box_h = ty + 12
    cv2.rectangle(frame, (0, fh - box_h), (fw, fh), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (10, fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # (Motion legend removed as requested)

    small = cv2.resize(frame, (640, 360))
    cv2.imshow('VC_1 - Single Road Vehicle Counter', small)

    # Wait for ESC; multiply delay to slow playback
    wait_ms = max(1, int(delay * play_speed))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == 27:  # ESC
        break
    # Runtime speed controls
    if key in (ord('+'), ord('=')):
        play_speed = min(play_speed + 0.5, 10)
        print(f"Playback slowed: play_speed={play_speed}")
    elif key == ord('-'):
        play_speed = max(1, play_speed - 0.5)
        print(f"Playback sped up: play_speed={play_speed}")
    elif key == ord('0'):
        play_speed = 1
        print("Playback reset: play_speed=1 (real-time)")

cap.release()
cv2.destroyAllWindows()

# Print final summary
print("\n" + "=" * 60)
print("VIDEO PROCESSING COMPLETE")
print("=" * 60)
print(f"Total Vehicles Counted: {vehicle_count}")
print("=" * 60 + "\n")
