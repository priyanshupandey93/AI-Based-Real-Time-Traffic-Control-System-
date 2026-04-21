import cv2
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ── Shared count store: bridges detector threads → simulation_dashboard ──
try:
    import count_store
    COUNT_STORE_AVAILABLE = True
except ImportError:
    COUNT_STORE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOTracker:
    def __init__(self, model):
        self.model = model
        self.CONFIDENCE_THRESHOLD = 0.40
        self.VEHICLE_CLASSES = [2, 3, 5, 7]
        self.MOTION_THRESHOLD = 5
        
        self.vehicle_count = 0
        self.tracked_vehicles = {}
        self.counted_ids = set()
        self.next_vehicle_id = 1
        self.MAX_MISSING_FRAMES = 15
        
    def process(self, frame):
        fh, fw = frame.shape[:2]
        count_line_position = int(fh * 0.65)
        
        results = self.model(frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        frame_dets = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                
                if class_id not in self.VEHICLE_CLASSES:
                    continue
                    
                wbox = x2 - x1
                hbox = y2 - y1
                
                if wbox < 60 or hbox < 60:
                    continue
                    
                cx = int(x1 + wbox / 2)
                cy = int(y1 + hbox / 2)
                
                if cx < int(fw * 0.15) or cx > int(fw * 0.85):
                    continue
                    
                frame_dets.append((x1, y1, wbox, hbox, cx, cy))

        current_detections = []
        new_tracked_vehicles = {}
        unassigned_dets = frame_dets.copy()
        
        for t_id, (tcx, tcy, missing) in sorted(self.tracked_vehicles.items(), key=lambda x: x[1][2]):
            if not unassigned_dets:
                if missing < self.MAX_MISSING_FRAMES:
                    new_tracked_vehicles[t_id] = (tcx, tcy, missing + 1)
                continue
                
            min_dist = float('inf')
            best_match_idx = -1
            
            for i, (x1, y1, wbox, hbox, cx, cy) in enumerate(unassigned_dets):
                dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
                    
            if best_match_idx != -1:
                x1, y1, wbox, hbox, cx, cy = unassigned_dets.pop(best_match_idx)
                new_tracked_vehicles[t_id] = (cx, cy, 0)
                current_detections.append((x1, y1, wbox, hbox, cx, cy, t_id, tcx, tcy))
            elif missing < self.MAX_MISSING_FRAMES:
                new_tracked_vehicles[t_id] = (tcx, tcy, missing + 1)
                
        for x1, y1, wbox, hbox, cx, cy in unassigned_dets:
            assigned_id = self.next_vehicle_id
            self.next_vehicle_id += 1
            new_tracked_vehicles[assigned_id] = (cx, cy, 0)
            current_detections.append((x1, y1, wbox, hbox, cx, cy, assigned_id, cx, cy))
            
        self.tracked_vehicles = new_tracked_vehicles

        for x, y, wbox, hbox, cx, cy, vid, last_cx, last_cy in current_detections:
            if (last_cy <= count_line_position and cy > count_line_position) or \
               (last_cy >= count_line_position and cy < count_line_position):
                if vid not in self.counted_ids:
                    self.vehicle_count += 1
                    self.counted_ids.add(vid)

        # Drawing
        for x, y, wbox, hbox, cx, cy, vid, last_cx, last_cy in current_detections:
            dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
            is_moving = dist >= self.MOTION_THRESHOLD
            color = (0, 255, 0) if is_moving else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + wbox, y + hbox), color, 2)
            cv2.putText(frame, f"VID {vid}", (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.line(frame, (25, count_line_position), (fw - 25, count_line_position), (255, 127, 0), 3)
        cv2.putText(frame, "Detection Line", (30, count_line_position - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 127, 0), 1)

        count_text = f"Total Vehicles Counted: {self.vehicle_count}"
        frame_padded = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv2.putText(frame_padded, count_text, (10, fh + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame_padded, count_text

    def get_direction_count(self):
        """Total crossing count for Camera 1 (North) and Camera 2 (South)."""
        return self.vehicle_count


class MOGTracker3:
    def __init__(self):
        try:
            self.algo = cv2.bgsegm.createBackgroundSubtractorMOG()
        except Exception:
            self.algo = cv2.createBackgroundSubtractorMOG2()
            
        self.min_width_rectangle = 50
        self.min_height_rectangle = 50
        self.IOU_WEIGHT = 1000
        self.MIN_HITS_TO_COUNT = 1
        
        self.next_vid = 1
        self.tracks = {}
        self.up_count = 0
        self.down_count = 0
        self.max_disappeared = 12
        self.max_match_distance = 80

    def reset_tracks(self):
        """Clear active track state on video loop.
        Keeps cumulative up_count / down_count intact.
        Without this, vehicles at the start of a looped video get
        matched to old tracks marked counted=True and are never counted.
        """
        self.tracks = {}

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea
        return interArea / union if union > 0 else 0

    def process(self, frame):
        fh, fw = frame.shape[:2]
        count_line_position = int(fh * 0.8)
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)

        fgmask = self.algo.apply(blur)
        _, thresh = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=3)
        clean = cv2.dilate(clean, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500: continue
            x, y, wbox, hbox = cv2.boundingRect(c)
            if wbox < self.min_width_rectangle or hbox < self.min_height_rectangle: continue
            aspect = float(wbox) / float(hbox) if hbox > 0 else 0
            if aspect < 0.4 or aspect > 5.0: continue
            cx = int(x + wbox / 2)
            cy = int(y + hbox / 2)
            detections.append((x, y, wbox, hbox, cx, cy))

        assigned_tracks = set()
        assigned_dets = set()
        if len(self.tracks) > 0 and len(detections) > 0:
            track_ids = list(self.tracks.keys())
            cost_list = []
            for vid in track_ids:
                tx, ty = self.tracks[vid]['centroid']
                tbbox = self.tracks[vid].get('bbox', (0, 0, 0, 0))
                for di, det in enumerate(detections):
                    x, y, wbox, hbox, cx, cy = det
                    d = (tx - cx) ** 2 + (ty - cy) ** 2
                    i = self.iou(tbbox, (x, y, wbox, hbox))
                    cost = d - (i * self.IOU_WEIGHT)
                    cost_list.append((cost, d, vid, di))
            cost_list.sort(key=lambda x: x[0])
            for cost, d, vid, di in cost_list:
                if vid in assigned_tracks or di in assigned_dets: continue
                if d > (self.max_match_distance ** 2): continue
                
                x, y, wbox, hbox, cx, cy = detections[di]
                prev_cx, prev_cy = self.tracks[vid]['centroid']
                self.tracks[vid]['centroid'] = (cx, cy)
                self.tracks[vid]['bbox'] = (x, y, wbox, hbox)
                self.tracks[vid]['disappeared'] = 0
                self.tracks[vid]['hits'] = self.tracks[vid].get('hits', 0) + 1
                
                prev_side = self.tracks[vid].get('side', 'above' if prev_cy < count_line_position else 'below')
                new_side = 'below' if cy > count_line_position else 'above'
                
                if not self.tracks[vid].get('counted', False) and prev_side != new_side and self.tracks[vid].get('hits', 0) >= self.MIN_HITS_TO_COUNT:
                    self.tracks[vid]['counted'] = True
                    self.tracks[vid]['flash'] = 10
                    if prev_side == 'above' and new_side == 'below':
                        self.down_count += 1
                    elif prev_side == 'below' and new_side == 'above':
                        self.up_count += 1
                        
                self.tracks[vid]['side'] = new_side
                assigned_tracks.add(vid)
                assigned_dets.add(di)

        for i, det in enumerate(detections):
            if i in assigned_dets: continue
            x, y, wbox, hbox, cx, cy = det
            vid = self.next_vid
            self.next_vid += 1
            side = 'below' if cy > count_line_position else 'above'
            self.tracks[vid] = {'centroid': (cx, cy), 'bbox': (x, y, wbox, hbox), 'disappeared': 0, 'counted': False, 'flash': 0, 'side': side, 'initial_side': side, 'hits': 1}
            
            if y <= count_line_position <= (y + hbox) and not self.tracks[vid]['counted']:
                frac_above = (count_line_position - y) / float(hbox)
                frac_below = ((y + hbox) - count_line_position) / float(hbox)
                if frac_below >= 0.6:
                    self.tracks[vid]['counted'] = True
                    self.tracks[vid]['flash'] = 10
                    self.down_count += 1
                elif frac_above >= 0.6:
                    self.tracks[vid]['counted'] = True
                    self.tracks[vid]['flash'] = 10
                    self.up_count += 1

        for vid in list(self.tracks.keys()):
            if vid in assigned_tracks: continue
            self.tracks[vid]['disappeared'] += 1
            cx, cy = self.tracks[vid]['centroid']
            margin = 50
            if cx < -margin or cx > fw + margin or cy < -margin or cy > fh + margin or self.tracks[vid]['disappeared'] > self.max_disappeared:
                if not self.tracks[vid].get('counted', False) and self.tracks[vid].get('initial_side') is not None and self.tracks[vid].get('side') is not None and self.tracks[vid]['initial_side'] != self.tracks[vid]['side']:
                    if self.tracks[vid]['initial_side'] == 'above' and self.tracks[vid]['side'] == 'below':
                        self.down_count += 1
                    elif self.tracks[vid]['initial_side'] == 'below' and self.tracks[vid]['side'] == 'above':
                        self.up_count += 1
                del self.tracks[vid]

        for vid, t in self.tracks.items():
            x, y, wbox, hbox = t['bbox']
            flash = t.get('flash', 0)
            counted = t.get('counted', False)
            if flash > 0:
                color = (0, 0, 255)
                t['flash'] = flash - 1
            elif counted:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + wbox, y + hbox), color, 2)
            cv2.putText(frame, f"VID {vid}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.line(frame, (25, count_line_position), (fw - 25, count_line_position), (255, 127, 0), 3)
        cv2.putText(frame, "Detection Line", (30, count_line_position - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 127, 0), 1)

        count_text = f"Up: {self.up_count}   Down: {self.down_count}   Total: {self.up_count + self.down_count}"
        frame_padded = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv2.putText(frame_padded, count_text, (10, fh + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame_padded, count_text

    def get_direction_count(self):
        """DOWN count only — westbound vehicles entering intersection (Camera 3)."""
        return self.down_count


class MOGTracker4:
    def __init__(self):
        try:
            self.algo = cv2.bgsegm.createBackgroundSubtractorMOG()
        except Exception:
            self.algo = cv2.createBackgroundSubtractorMOG2()

        # Reduced to 50x50 to match MOGTracker3 — 80x80 was too large
        # for the 640x360 display frame and suppressed most detections
        self.min_width_rectangle  = 50
        self.min_height_rectangle = 50

        self.next_vid         = 1
        self.tracks           = {}
        self.up_count         = 0
        self.down_count       = 0
        self.max_disappeared  = 10
        self.max_match_distance = 80

    def reset_tracks(self):
        """Clear active track state on video loop.
        Keeps cumulative up_count / down_count intact.
        Without this, vehicles at the start of a looped video get
        matched to old tracks marked counted=True and are never counted.
        """
        self.tracks = {}

    def process(self, frame):
        fh, fw = frame.shape[:2]
        count_line_position = int(fh * 0.8)
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)

        fgmask = self.algo.apply(blur)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean = cv2.dilate(clean, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 500: continue
            x, y, wbox, hbox = cv2.boundingRect(c)
            if wbox < self.min_width_rectangle or hbox < self.min_height_rectangle: continue
            aspect = float(wbox) / float(hbox) if hbox > 0 else 0
            if aspect < 0.4 or aspect > 5.0: continue
            cx = int(x + wbox / 2)
            cy = int(y + hbox / 2)
            detections.append((x, y, wbox, hbox, cx, cy))

        assigned_tracks = set()
        assigned_dets = set()
        for i, det in enumerate(detections):
            x, y, wbox, hbox, cx, cy = det
            best_vid = None
            best_dist = None
            for vid, t in self.tracks.items():
                if vid in assigned_tracks: continue
                tx, ty = t['centroid']
                dist = (tx - cx) ** 2 + (ty - cy) ** 2
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_vid = vid
            if best_vid is not None and best_dist is not None and best_dist <= (self.max_match_distance ** 2):
                prev_cx, prev_cy = self.tracks[best_vid]['centroid']
                self.tracks[best_vid]['centroid'] = (cx, cy)
                self.tracks[best_vid]['bbox'] = (x, y, wbox, hbox)
                self.tracks[best_vid]['disappeared'] = 0
                if not self.tracks[best_vid].get('counted', False):
                    if (prev_cy < count_line_position <= cy):
                        self.tracks[best_vid]['counted'] = True
                        self.tracks[best_vid]['flash'] = 10
                        self.down_count += 1
                    elif (prev_cy > count_line_position >= cy):
                        self.tracks[best_vid]['counted'] = True
                        self.tracks[best_vid]['flash'] = 10
                        self.up_count += 1
                self.tracks[best_vid].setdefault('counted', False)
                assigned_tracks.add(best_vid)
                assigned_dets.add(i)

        for i, det in enumerate(detections):
            if i in assigned_dets: continue
            x, y, wbox, hbox, cx, cy = det
            vid = self.next_vid
            self.next_vid += 1
            self.tracks[vid] = {'centroid': (cx, cy), 'bbox': (x, y, wbox, hbox), 'disappeared': 0, 'counted': False, 'flash': 0}

        for vid in list(self.tracks.keys()):
            if vid in assigned_tracks: continue
            self.tracks[vid]['disappeared'] += 1
            cx, cy = self.tracks[vid]['centroid']
            margin = 50
            if cx < -margin or cx > fw + margin or cy < -margin or cy > fh + margin or self.tracks[vid]['disappeared'] > self.max_disappeared:
                del self.tracks[vid]

        for vid, t in self.tracks.items():
            x, y, wbox, hbox = t['bbox']
            flash = t.get('flash', 0)
            counted = t.get('counted', False)
            if flash > 0:
                color = (0, 0, 255)
                t['flash'] = flash - 1
            elif counted:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + wbox, y + hbox), color, 2)
            cv2.putText(frame, f"VID {vid}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.line(frame, (25, count_line_position), (fw - 25, count_line_position), (255, 127, 0), 3)
        cv2.putText(frame, "Detection Line", (30, count_line_position - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 127, 0), 1)

        count_text = f"Up: {self.up_count}   Down: {self.down_count}   Total: {self.up_count + self.down_count}"
        frame_padded = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv2.putText(frame_padded, count_text, (10, fh + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame_padded, count_text

    def get_direction_count(self):
        """DOWN count only — eastbound vehicles entering intersection (Camera 4)."""
        return self.down_count


class VideoStreamThread(threading.Thread):
    def __init__(self, video_path, tracker, direction="Unknown"):
        super().__init__()
        self.video_path = video_path
        self.tracker = tracker
        self.direction = direction   # "North", "South", "West", or "East"
        self.cap = cv2.VideoCapture(self.video_path)
        self.current_frame = None
        self.current_count_text = "Loading..."
        self.running = True
        self.lock = threading.Lock()
        self._last_pushed_count = -1   # track last value sent to count_store
        self._count_baseline    = 0    # subtracted from raw tracker total after a reset

        # Calculate target delay based on fps, add 3x multiplier to mimic VC_3.py and VC_4.py slow playback
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.target_delay = (1.0 / fps) * 3
        
    def run(self):
        while self.running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # Video ended — loop back to start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Clear active tracks so vehicles at the start of the
                # next loop aren't matched to old counted=True tracks
                if hasattr(self.tracker, 'reset_tracks'):
                    self.tracker.reset_tracks()
                continue
            
            # Reduce resolution to keep Tkinter fast
            frame = cv2.resize(frame, (640, 360))
            
            # Process frame
            processed_frame, count_text = self.tracker.process(frame)
            
            # Convert to RGB for PIL
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            with self.lock:
                self.current_frame = rgb_frame
                self.current_count_text = count_text

            # Push to shared store — check for reset request first
            if COUNT_STORE_AVAILABLE:
                raw = self.tracker.get_direction_count()

                # If simulation requested a reset for this lane, re-baseline now
                if count_store.consume_reset_flag(self.direction):
                    self._count_baseline = raw   # new zero-point
                    self._last_pushed_count = -1  # force an immediate write of 0

                adjusted = max(0, raw - self._count_baseline)
                if adjusted != self._last_pushed_count:
                    count_store.update_count(self.direction, adjusted)
                    self._last_pushed_count = adjusted

            process_time = time.time() - start_time
            sleep_time = max(0.01, self.target_delay - process_time)
            time.sleep(sleep_time)
            
    def get_latest_data(self):
        with self.lock:
            return self.current_frame, self.current_count_text
            
    def stop(self):
        self.running = False
        self.cap.release()

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Counting Dashboard")
        self.root.geometry("1050x700")  # Reduced sizing to fit all screens
        self.root.configure(bg='#1e1e1e')
        
        # Top Title
        title_lbl = tk.Label(self.root, text="YOLO & Background Subtraction Multi-Stream Vehicle Counter", 
                             font=("Helvetica", 20, "bold"), bg='#1e1e1e', fg='white')
        title_lbl.pack(pady=10)
        
        self.main_frame = tk.Frame(self.root, bg='#1e1e1e')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Configure grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        self.video_panels = []
        
        # Headers & Setup Grid
        titles = ["Camera 1 North", "Camera 2 South", 
                  "Camera 3 West", "Camera 4 East"]
                  
        grid_positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for i in range(4):
            frame = tk.Frame(self.main_frame, bg='#2d2d2d', bd=2, relief=tk.RIDGE)
            frame.grid(row=grid_positions[i][0], column=grid_positions[i][1], padx=10, pady=10, sticky="nsew")
            
            lbl_title = tk.Label(frame, text=titles[i], font=("Helvetica", 14), bg='#2d2d2d', fg='#4aa8ff')
            lbl_title.pack(pady=5)
            
            panel = tk.Label(frame, bg='black')
            panel.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
            self.video_panels.append(panel)

        # Load YOLO Model Once for VC_1 and VC_2 to save VRAM
        if YOLO_AVAILABLE:
            print("Loading YOLOv8 Nano model...")
            self.yolo_model = YOLO('yolov8n.pt')
        else:
            print("YOLO NOT FOUND!")
            self.yolo_model = None

        # Setup Threads
        self.threads = []
        video_files = ['test_videos/1.mp4', 'test_videos/2.mp4', 'test_videos/3.mp4', 'test_videos/4.mp4']
        
        # Tracker assignments
        trackers = [
            YOLOTracker(self.yolo_model) if self.yolo_model else MOGTracker4(),
            YOLOTracker(self.yolo_model) if self.yolo_model else MOGTracker4(),
            MOGTracker3(),
            MOGTracker4()
        ]
        
        directions = ["North", "South", "West", "East"]
        for i in range(4):
            t = VideoStreamThread(video_files[i], trackers[i], directions[i])
            self.threads.append(t)
            t.daemon = True
            t.start()

        # Start Tkinter Loop Update
        self.update_gui()
        
    def update_gui(self):
        for i, thread in enumerate(self.threads):
            frame, count_text = thread.get_latest_data()
            if frame is not None:
                # Resize for display to fit smaller screens
                # Using PIL to display
                img = Image.fromarray(frame)
                img = img.resize((480, 300))  # Aspect ratio accommodates height padding
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_panels[i].imgtk = imgtk  # keep reference
                self.video_panels[i].configure(image=imgtk)

        self.root.after(30, self.update_gui)

    def on_closing(self):
        print("Stopping threads...")
        for t in self.threads:
            t.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
