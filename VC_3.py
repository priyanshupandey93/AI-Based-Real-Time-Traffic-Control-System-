import cv2
import numpy as np
import time

# Video Capture
cap = cv2.VideoCapture('test_videos/3.mp4')

min_width_rectangle = 50  # min width of rectangle
min_height_rectangle = 50  # min height of rectangle
IOU_WEIGHT = 1000
MIN_HITS_TO_COUNT = 1
IOU_THRESHOLD = 0.15


# Initialize Subtractor (use bgsegm if available, else fall back to MOG2)
try:
    algo = cv2.bgsegm.createBackgroundSubtractorMOG()
except Exception:
    algo = cv2.createBackgroundSubtractorMOG2()
    print("cv2.bgsegm not found — falling back to MOG2.\n" \
          "To use bgsegm, install opencv-contrib-python: pip install opencv-contrib-python")
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if metadata missing
delay = int(1000 / fps)                # ms per frame

cv2.namedWindow('VC_3', cv2.WINDOW_NORMAL)
cv2.resizeWindow('VC_3', 640, 360)
print("Controls: '+' increase speed (slower), '-' decrease speed (faster), '0' reset to real-time")

# Tracker state
next_vid = 1
tracks = {}  # vid -> {centroid:(x,y), bbox:(x,y,w,h), disappeared:int, counted:bool, flash:int}
up_count = 0
down_count = 0
max_disappeared = 12
max_match_distance = 80
# Playback speed multiplier: 1 = real-time, >1 = slower playback (helps avoid dropped counts)
play_speed = 3  # higher -> slower playback (multiplies inter-frame delay)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    fh, fw = frame.shape[:2]
    if 'count_line_position' not in globals():
        count_line_position = int(fh * 0.8)

    # foreground mask -> clean -> contours
    fgmask = algo.apply(blur)
    _, thresh = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=3)
    clean = cv2.dilate(clean, np.ones((7, 7), np.uint8), iterations=2)
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

    # matching detections to existing tracks using greedy smallest-distance matching
    assigned_tracks = set()
    assigned_dets = set()
    if len(tracks) > 0 and len(detections) > 0:
        track_ids = list(tracks.keys())
        # build cost list using distance minus IoU weight (prefer matches with high IoU)
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]
            union = boxAArea + boxBArea - interArea
            return interArea / union if union > 0 else 0

        cost_list = []
        for vid in track_ids:
            tx, ty = tracks[vid]['centroid']
            tbbox = tracks[vid].get('bbox', (0, 0, 0, 0))
            for di, det in enumerate(detections):
                x, y, wbox, hbox, cx, cy = det
                d = (tx - cx) ** 2 + (ty - cy) ** 2
                i = iou(tbbox, (x, y, wbox, hbox))
                # lower cost = better match; penalize distance, reward IoU
                cost = d - (i * IOU_WEIGHT)
                cost_list.append((cost, d, vid, di))
        cost_list.sort(key=lambda x: x[0])
        for cost, d, vid, di in cost_list:
            if vid in assigned_tracks or di in assigned_dets:
                continue
            # d is squared distance. enforce a squared threshold to avoid unit mismatch
            if d > (max_match_distance ** 2):
                continue
            # assign detection di to track vid
            x, y, wbox, hbox, cx, cy = detections[di]
            prev_cx, prev_cy = tracks[vid]['centroid']
            # jump is squared distance (d). compare against squared thresholds.
            jump = d
            max_jump = (max_match_distance ** 2)
            if jump > max_jump:
                continue
            tracks[vid]['centroid'] = (cx, cy)
            tracks[vid]['bbox'] = (x, y, wbox, hbox)
            tracks[vid]['disappeared'] = 0
            # increment hits (stabilization) for confirmed tracking
            tracks[vid]['hits'] = tracks[vid].get('hits', 0) + 1
            # crossing detection based on side change
            prev_side = tracks[vid].get('side', 'above' if prev_cy < count_line_position else 'below')
            new_side = 'below' if cy > count_line_position else 'above'
            # only count when the object has been seen for a couple frames (reduce false/missed counts)
            if not tracks[vid].get('counted', False) and prev_side != new_side and tracks[vid].get('hits', 0) >= MIN_HITS_TO_COUNT:
                tracks[vid]['counted'] = True
                tracks[vid]['flash'] = 10
                if prev_side == 'above' and new_side == 'below':
                    down_count += 1
                    print(f"Counted DOWN VID {vid} — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")
                elif prev_side == 'below' and new_side == 'above':
                    up_count += 1
                    print(f"Counted UP VID {vid} — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")
            tracks[vid]['side'] = new_side
            tracks[vid].setdefault('counted', False)
            tracks[vid].setdefault('flash', 0)
            assigned_tracks.add(vid)
            assigned_dets.add(di)

    # create tracks for unassigned detections
    for i, det in enumerate(detections):
        if i in assigned_dets:
            continue
        x, y, wbox, hbox, cx, cy = det
        vid = next_vid
        next_vid += 1
        side = 'below' if cy > count_line_position else 'above'
        # record the initial side so we can detect missed crossings if the track disappears
        tracks[vid] = {'centroid': (cx, cy), 'bbox': (x, y, wbox, hbox), 'disappeared': 0, 'counted': False, 'flash': 0, 'side': side, 'initial_side': side, 'hits': 1}
        # Conservative immediate count: if bbox overlaps the count line and
        # a majority of the bbox lies on one side, count immediately.
        if y <= count_line_position <= (y + hbox) and not tracks[vid]['counted']:
            frac_above = (count_line_position - y) / float(hbox)
            frac_below = ((y + hbox) - count_line_position) / float(hbox)
            if frac_below >= 0.6:
                tracks[vid]['counted'] = True
                tracks[vid]['flash'] = 10
                down_count += 1
                print(f"Counted DOWN VID {vid} — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")
            elif frac_above >= 0.6:
                tracks[vid]['counted'] = True
                tracks[vid]['flash'] = 10
                up_count += 1
                print(f"Counted UP VID {vid} — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")

    # increase disappeared for unassigned tracks and remove if outside frame
    for vid in list(tracks.keys()):
        if vid in assigned_tracks:
            continue
        tracks[vid]['disappeared'] += 1
        cx, cy = tracks[vid]['centroid']
        margin = 50
        if cx < -margin or cx > fw + margin or cy < -margin or cy > fh + margin:
            # before removing, if this track wasn't counted but its side changed from initial, count it
            if not tracks[vid].get('counted', False) and tracks[vid].get('initial_side') is not None and tracks[vid].get('side') is not None and tracks[vid]['initial_side'] != tracks[vid]['side']:
                # determine direction
                if tracks[vid]['initial_side'] == 'above' and tracks[vid]['side'] == 'below':
                    down_count += 1
                    print(f"(Late) Counted DOWN VID {vid} on disappearance — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")
                elif tracks[vid]['initial_side'] == 'below' and tracks[vid]['side'] == 'above':
                    up_count += 1
                    print(f"(Late) Counted UP VID {vid} on disappearance — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")
            del tracks[vid]
            continue
        if tracks[vid]['disappeared'] > max_disappeared:
            # before removing due to timeout, try to catch missed crossing
            if not tracks[vid].get('counted', False) and tracks[vid].get('initial_side') is not None and tracks[vid].get('side') is not None and tracks[vid]['initial_side'] != tracks[vid]['side']:
                if tracks[vid]['initial_side'] == 'above' and tracks[vid]['side'] == 'below':
                    down_count += 1
                    print(f"(Late) Counted DOWN VID {vid} on timeout — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")
                elif tracks[vid]['initial_side'] == 'below' and tracks[vid]['side'] == 'above':
                    up_count += 1
                    print(f"(Late) Counted UP VID {vid} on timeout — Down:{down_count} Up:{up_count} Total:{up_count+down_count}")
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
    cv2.imshow('VC_3', small)

    # wait for ESC; multiply delay to slow playback and reduce frame drops
    wait_ms = max(1, int(delay * play_speed))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == 27:  # ESC
        break
    # runtime speed controls: '+' or '=' to increase (slower), '-' to decrease (faster), '0' to reset
    if key in (ord('+'), ord('=')):
        play_speed = min(play_speed + 1, 10)
        print(f"Playback slowed: play_speed={play_speed}")
    elif key == ord('-'):
        play_speed = max(1, play_speed - 1)
        print(f"Playback sped up: play_speed={play_speed}")
    elif key == ord('0'):
        play_speed = 1
        print("Playback reset: play_speed=1 (real-time)")

cap.release()
cv2.destroyAllWindows()
