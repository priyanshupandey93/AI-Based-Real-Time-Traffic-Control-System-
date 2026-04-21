import cv2
import numpy as np
import time

# Video Capture
cap = cv2.VideoCapture('test_videos/4.mp4')

min_width_rectangle = 80  # min width of rectangle
min_height_rectangle = 80  # min height of rectangle


# Initialize Subtractor (use bgsegm if available, else fall back to MOG2)
try:
    algo = cv2.bgsegm.createBackgroundSubtractorMOG()
except Exception:
    algo = cv2.createBackgroundSubtractorMOG2()
    print("cv2.bgsegm not found — falling back to MOG2.\n" \
          "To use bgsegm, install opencv-contrib-python: pip install opencv-contrib-python")
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if metadata missing
delay = int(1000 / fps)                # ms per frame

cv2.namedWindow('Original Video 4', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Video 4', 640, 360)

# Tracker state
next_vid = 1
tracks = {}  # vid -> {centroid:(x,y), bbox:(x,y,w,h), disappeared:int, counted:bool, flash:int}
up_count = 0
down_count = 0
max_disappeared = 10
max_match_distance = 80

frame_idx = 0
print("\n" + "="*50)
print("VEHICLE COUNTER - TERMINAL OUTPUT")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    prev_up_count = up_count
    prev_down_count = down_count
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    fh, fw = frame.shape[:2]
    if 'count_line_position' not in globals():
        count_line_position = int(fh * 0.8)

    # foreground mask -> clean -> contours
    fgmask = algo.apply(blur)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.dilate(clean, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []  # list of (x,y,w,h,cx,cy)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, wbox, hbox = cv2.boundingRect(c)
        if wbox < min_width_rectangle or hbox < min_height_rectangle:
            continue
        aspect = float(wbox) / float(hbox) if hbox > 0 else 0
        if aspect < 0.4 or aspect > 5.0:
            continue
        rect_area = wbox * hbox
        if rect_area <= 0:
            continue
        extent = area / float(rect_area)
        if extent < 0.25:
            continue
        cx = int(x + wbox / 2)
        cy = int(y + hbox / 2)
        detections.append((x, y, wbox, hbox, cx, cy))

    # matching detections to existing tracks (greedy nearest)
    assigned_tracks = set()
    assigned_dets = set()
    for i, det in enumerate(detections):
        x, y, wbox, hbox, cx, cy = det
        best_vid = None
        best_dist = None
        for vid, t in tracks.items():
            if vid in assigned_tracks:
                continue
            tx, ty = t['centroid']
            dist = (tx - cx) ** 2 + (ty - cy) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_vid = vid
        if best_vid is not None and best_dist is not None and best_dist <= (max_match_distance ** 2):
            # assign
            prev_cx, prev_cy = tracks[best_vid]['centroid']
            tracks[best_vid]['centroid'] = (cx, cy)
            tracks[best_vid]['bbox'] = (x, y, wbox, hbox)
            tracks[best_vid]['disappeared'] = 0
            # crossing detection for up/down
            if not tracks[best_vid].get('counted', False):
                if (prev_cy < count_line_position <= cy):
                    tracks[best_vid]['counted'] = True
                    tracks[best_vid]['flash'] = 10
                    down_count += 1
                    print(f"[Frame {frame_idx}] Vehicle {best_vid} counted DOWNWARD | Up: {up_count}  Down: {down_count}  Total: {up_count + down_count}")
                elif (prev_cy > count_line_position >= cy):
                    tracks[best_vid]['counted'] = True
                    tracks[best_vid]['flash'] = 10
                    up_count += 1
                    print(f"[Frame {frame_idx}] Vehicle {best_vid} counted UPWARD   | Up: {up_count}  Down: {down_count}  Total: {up_count + down_count}")
            tracks[best_vid].setdefault('counted', False)
            tracks[best_vid].setdefault('flash', 0)
            assigned_tracks.add(best_vid)
            assigned_dets.add(i)

    # create tracks for unassigned detections
    for i, det in enumerate(detections):
        if i in assigned_dets:
            continue
        x, y, wbox, hbox, cx, cy = det
        vid = next_vid
        next_vid += 1
        tracks[vid] = {'centroid': (cx, cy), 'bbox': (x, y, wbox, hbox), 'disappeared': 0, 'counted': False, 'flash': 0}

    # increase disappeared for unassigned tracks and remove if outside frame
    for vid in list(tracks.keys()):
        if vid in assigned_tracks:
            continue
        tracks[vid]['disappeared'] += 1
        cx, cy = tracks[vid]['centroid']
        margin = 50
        if cx < -margin or cx > fw + margin or cy < -margin or cy > fh + margin:
            del tracks[vid]
            continue
        if tracks[vid]['disappeared'] > max_disappeared:
            del tracks[vid]

    # draw tracked boxes and IDs (flash color for recently counted)
    for vid, t in tracks.items():
        x, y, wbox, hbox = t['bbox']
        flash = t.get('flash', 0)
        counted = t.get('counted', False)
        if flash and flash > 0:
            color = (0, 0, 255)
            t['flash'] = flash - 1
        elif counted:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + wbox, y + hbox), color, 2)
        cv2.putText(frame, f"VID {vid}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # draw count line
    cv2.line(frame, (25, count_line_position), (fw - 25, count_line_position), (255, 127, 0), 3)

    # show up/down/total counts at bottom
    total_count = up_count + down_count
    count_text = f"Up: {up_count}   Down: {down_count}   Total: {total_count}"
    (tx, ty), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    box_h = ty + 12
    cv2.rectangle(frame, (0, fh - box_h), (fw, fh), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (10, fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    small = cv2.resize(frame, (640, 360))
    cv2.imshow('Original Video 4', small)

    # wait for ESC
    if cv2.waitKey(delay) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print final summary
print("\n" + "="*50)
print("VIDEO PROCESSING COMPLETE")
print("="*50)
print(f"Total Vehicles Counted Up:     {up_count}")
print(f"Total Vehicles Counted Down:   {down_count}")
print(f"TOTAL VEHICLES:                {up_count + down_count}")
print("="*50 + "\n")
