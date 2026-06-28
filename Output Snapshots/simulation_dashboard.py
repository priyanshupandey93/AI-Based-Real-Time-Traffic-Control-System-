"""
simulation_dashboard.py
──────────────────────────────────────────────────────────────
Adaptive Traffic Signal Simulation Dashboard

• Reads live vehicle counts from count_store.py every 500 ms
  (written in real-time by main_dashboard.py detectors).
• Displays auto-updating live counts for all 4 camera lanes.
• On "Start Simulation" — snapshots the current counts,
  runs the density-based time-allocation algorithm, and
  cycles through the 4 traffic signals with countdown timers.

Algorithm:  Gi = min(CAP, 10 + Di)
  Normal   → cap 60 s per lane
  Extreme  → priority lane cap 90 s, others cap 60 s
"""

import tkinter as tk
from tkinter import font as tkfont

# ── Shared count store ────────────────────────────────────────────────────
try:
    import count_store
    COUNT_STORE_AVAILABLE = True
except ImportError:
    COUNT_STORE_AVAILABLE = False
    print("[WARNING] count_store.py not found – counts will stay at 0.")

# ─────────────── Algorithm Constants ──────────────────────────────────────
MIN_GREEN    = 10    # seconds – starvation prevention
TOTAL_CYCLE  = 60    # seconds – normal per-lane cap
PRIORITY_MAX = 90    # seconds – extreme congestion priority cap

CAMERAS = [
    ("Camera 1", "North"),
    ("Camera 2", "South"),
    ("Camera 3", "West"),
    ("Camera 4", "East"),
]

# ─────────────── Colour Palette ───────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
CARD     = "#21262d"
GREEN_ON = "#3fb950"
RED_ON   = "#f85149"
AMBER    = "#e3b341"
CYAN     = "#58a6ff"
WHITE    = "#f0f6fc"
GREY     = "#8b949e"
BORDER   = "#30363d"


# ─────────────── Pure Algorithm Functions ─────────────────────────────────

def _is_extreme(densities: list) -> bool:
    """True when the top lane has ≥ 2.5× more vehicles than second-highest."""
    sd = sorted(densities, reverse=True)
    if len(sd) < 2 or sd[1] == 0:
        return False
    return sd[0] >= 2.5 * sd[1]


def allocate_time(densities: list) -> tuple:
    """
    Gi = min(CAP, max(MIN_GREEN, MIN_GREEN + Di))
       = 10 s base + 1 s per vehicle, capped per case.

    Returns (times_list, mode_label_string).
    """
    total = sum(densities)

    if total == 0:
        return [float(MIN_GREEN)] * 4, "Equal – No Traffic  (10 s each)"

    if _is_extreme(densities):
        max_idx = densities.index(max(densities))
        times = []
        for i, d in enumerate(densities):
            if i == max_idx:
                times.append(float(min(PRIORITY_MAX, MIN_GREEN + d)))
            else:
                times.append(float(min(TOTAL_CYCLE, max(MIN_GREEN, MIN_GREEN + d))))
        cam, dr = CAMERAS[max_idx]
        return times, f"⚡ Extreme  –  Priority: {cam} ({dr})"

    times = [float(min(TOTAL_CYCLE, max(MIN_GREEN, MIN_GREEN + d))) for d in densities]
    return times, "✅ Normal  –  10 s base + 1 s per vehicle"


# ─────────────── Main Application ──────────────────────────────────────────

class SimulationDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Traffic Signal Simulation Dashboard")
        self.root.geometry("820x660")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        # Runtime state
        self.running       = False
        self.times         = [0.0] * 4
        self.live_counts   = [0]   * 4
        self.current_lane  = 0
        self.remaining     = 0
        self._tick_job     = None
        self._poll_job     = None
        # Tracks which lanes have already had (or are currently on) their timer,
        # so their count display is frozen at the snapshot value.
        self.frozen_lanes: set = set()

        # UI widget references
        self.count_labels  = {}
        self.sig_canvas    = {}
        self.sig_light     = {}
        self.sig_text      = {}
        self.timer_labels  = {}
        self.alloc_labels  = {}

        self._build_ui()
        self._poll_counts()   # start live count polling

    # ─────────── UI Construction ──────────────────────────────────────────

    def _build_ui(self):
        H1  = tkfont.Font(family="Helvetica", size=17, weight="bold")
        H2  = tkfont.Font(family="Helvetica", size=11, weight="bold")
        H3  = tkfont.Font(family="Helvetica", size=10)
        TMR = tkfont.Font(family="Helvetica", size=26, weight="bold")
        SM  = tkfont.Font(family="Helvetica", size=9)
        SIG = tkfont.Font(family="Helvetica", size=9, weight="bold")

        # ── Header ──────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=PANEL, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🚦  Adaptive Traffic Signal Simulation",
                 font=H1, bg=PANEL, fg=CYAN).pack()
        tk.Label(hdr, text="Real-Time Density-Based Signal Timing  |  4-Road Intersection",
                 font=SM, bg=PANEL, fg=GREY).pack(pady=(2, 0))

        # ── Unified table (headers + 4 data rows in one grid) ───────────────
        # Column widths in pixels — same for header row and every data row
        COL_W = [220, 130, 100, 140, 130]   # cam, count, signal, timer, alloc
        PAD_X = 8

        table = tk.Frame(self.root, bg=BG)
        table.pack(fill="x", padx=24, pady=(8, 0))

        for col, w in enumerate(COL_W):
            table.columnconfigure(col, minsize=w, weight=0)

        # ── Header row (row 0) ────────────────────────────────────────────
        headers = ["Camera / Direction", "Live Count", "Signal", "Countdown", "Alloc. Time"]
        for col, txt in enumerate(headers):
            tk.Label(table, text=txt, font=H2, bg=BG, fg=CYAN,
                     anchor="center").grid(row=0, column=col,
                                           ipadx=PAD_X, ipady=6, sticky="ew")

        # ── Thin separator beneath headers ────────────────────────────────
        sep = tk.Frame(table, bg=BORDER, height=1)
        sep.grid(row=1, column=0, columnspan=5, sticky="ew", padx=4, pady=(0, 4))

        # ── Data rows (rows 2–5) ──────────────────────────────────────────
        for i, (cam, direction) in enumerate(CAMERAS):
            r = i + 2   # skip header row + separator

            # Camera / Direction label
            tk.Label(table,
                     text=f"{cam}  —  {direction}",
                     font=H2, bg=BG, fg=WHITE,
                     anchor="center").grid(row=r, column=0,
                                           ipadx=PAD_X, ipady=10, sticky="ew")

            # Live count (auto-updating, large amber number)
            cnt = tk.Label(table, text="0",
                           font=tkfont.Font(family="Helvetica", size=22, weight="bold"),
                           bg=BG, fg=AMBER, anchor="center")
            cnt.grid(row=r, column=1, ipadx=PAD_X, ipady=4, sticky="ew")
            self.count_labels[i] = cnt

            # Signal light canvas — centred inside its column
            sig_frame = tk.Frame(table, bg=BG)
            sig_frame.grid(row=r, column=2, sticky="ew", ipadx=0, ipady=6)
            sig_frame.columnconfigure(0, weight=1)

            cv = tk.Canvas(sig_frame, width=84, height=38,
                           bg=BG, highlightthickness=0)
            cv.grid(row=0, column=0)
            light = cv.create_oval(8, 4, 60, 34, fill=RED_ON, outline="")
            text  = cv.create_text(34, 19, text="RED", fill=WHITE, font=SIG)
            self.sig_canvas[i] = cv
            self.sig_light[i]  = light
            self.sig_text[i]   = text

            # Countdown timer
            t_lbl = tk.Label(table, text="──",
                             font=TMR, bg=BG, fg=GREY, anchor="center")
            t_lbl.grid(row=r, column=3, ipadx=PAD_X, ipady=4, sticky="ew")
            self.timer_labels[i] = t_lbl

            # Allocated time
            a_lbl = tk.Label(table, text="──",
                             font=H2, bg=BG, fg=GREY, anchor="center")
            a_lbl.grid(row=r, column=4, ipadx=PAD_X, ipady=4, sticky="ew")
            self.alloc_labels[i] = a_lbl

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(8, 0))

        # ── Status bar ───────────────────────────────────────────────────
        self.status_var = tk.StringVar(
            value="⏸  Live counts updating …  Click  ▶  to start the signal cycle")
        tk.Label(self.root, textvariable=self.status_var,
                 font=H3, bg=PANEL, fg=AMBER,
                 anchor="center", pady=8).pack(fill="x")

        # ── Mode label ───────────────────────────────────────────────────
        self.mode_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.mode_var,
                 font=H3, bg=BG, fg=GREEN_ON, anchor="center").pack(pady=(6, 2))

        # ── Buttons ──────────────────────────────────────────────────────
        btns = tk.Frame(self.root, bg=BG)
        btns.pack(pady=10)

        self.start_btn = tk.Button(
            btns, text="▶   Start Simulation",
            font=H2, bg=CYAN, fg="#000000",
            relief="flat", padx=22, pady=8,
            activebackground=GREEN_ON, cursor="hand2",
            command=self.start_simulation
        )
        self.start_btn.grid(row=0, column=0, padx=14)

        self.stop_btn = tk.Button(
            btns, text="■   Stop",
            font=H2, bg=RED_ON, fg=WHITE,
            relief="flat", padx=22, pady=8,
            activebackground="#7a0000", cursor="hand2",
            command=self.stop_simulation
        )
        self.stop_btn.grid(row=0, column=1, padx=14)

        # ── Formula note ─────────────────────────────────────────────────
        tk.Label(
            self.root,
            text=("Formula:  Gi = 10 s (base) + 1 s per vehicle  |  "
                  "Min = 10 s  |  Normal cap = 60 s  |  Extreme priority cap = 90 s"),
            font=SM, bg=BG, fg=GREY, wraplength=780, justify="center"
        ).pack(pady=(6, 4))

        # ── Allocation log ───────────────────────────────────────────────
        log_wrap = tk.Frame(self.root, bg=CARD)
        log_wrap.pack(fill="x", padx=28, pady=(4, 14))
        tk.Label(log_wrap, text=" Allocation Log", font=SM,
                 bg=CARD, fg=GREY, anchor="w").pack(fill="x", padx=4, pady=(4, 0))
        self.log = tk.Text(log_wrap, height=4, bg=CARD, fg=GREY,
                           font=tkfont.Font(family="Courier", size=9),
                           relief="flat", state="disabled",
                           insertbackground=WHITE)
        self.log.pack(fill="x", padx=4, pady=(0, 6))

    # ─────────── Live Count Polling ───────────────────────────────────────

    def _poll_counts(self):
        """Read count_store every 200 ms and refresh the count labels.

        Lanes whose timer has already started (frozen_lanes) keep showing the
        snapshot count that was used for the algorithm so the display is stable
        while the signal is running.  All other lanes update normally.
        """
        if COUNT_STORE_AVAILABLE:
            data = count_store.get_all()
            dirs = ["North", "South", "West", "East"]
            for i, d in enumerate(dirs):
                val = max(0, int(data.get(d, 0)))
                if i not in self.frozen_lanes:
                    # Lane not yet served — keep live count updated
                    self.live_counts[i] = val
                    self.count_labels[i].config(text=str(val))
                # else: lane is frozen — leave count_labels and live_counts as-is

        self._poll_job = self.root.after(200, self._poll_counts)  # poll every 200 ms

    # ─────────── Signal Helpers ───────────────────────────────────────────

    def _set_green(self, i: int):
        self.sig_canvas[i].itemconfig(self.sig_light[i], fill=GREEN_ON)
        self.sig_canvas[i].itemconfig(self.sig_text[i],  text="GREEN", fill="#000000")
        self.timer_labels[i].config(fg=GREEN_ON)

    def _set_red(self, i: int):
        self.sig_canvas[i].itemconfig(self.sig_light[i], fill=RED_ON)
        self.sig_canvas[i].itemconfig(self.sig_text[i],  text="RED",   fill=WHITE)
        self.timer_labels[i].config(text="──", fg=GREY)

    def _reset_all_red(self, clear_alloc: bool = False):
        """Set all signals to RED.  Alloc. Time labels are only cleared when
        clear_alloc=True (i.e. at simulation start / stop), never mid-cycle."""
        for j in range(4):
            self._set_red(j)
            if clear_alloc:
                self.alloc_labels[j].config(text="──", fg=GREY)

    # ─────────── Log Helper ───────────────────────────────────────────────

    def _log(self, msg: str):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ─────────── Simulation Control ───────────────────────────────────────

    def start_simulation(self):
        if self.running:
            self.stop_simulation()

        # ── No global snapshot here ──────────────────────────────────────
        # Each lane will read its OWN live count when its turn arrives and
        # calculate its allocated time at that moment.
        self.times        = [0.0] * 4
        self.mode_var.set("Mode: ⏳ Dynamic — per-lane allocation on arrival")

        # Mark all alloc labels as pending
        for i in range(4):
            self.alloc_labels[i].config(text="pending…", fg=GREY)

        self._log("─" * 54)
        self._log("  Simulation started — allocating per lane on arrival")
        self._log("─" * 54)
        print("\n" + "=" * 56)
        print("  SIMULATION STARTED — per-lane dynamic allocation")
        print("=" * 56)

        # Reset signals, keep alloc labels (they show 'pending')
        self._reset_all_red(clear_alloc=False)
        self.frozen_lanes  = set()   # no lane frozen yet
        self.running       = True
        self.current_lane  = 0
        self._run_lane()

    def stop_simulation(self):
        self.running = False
        if self._tick_job:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None
        self.frozen_lanes = set()   # unfreeze all lanes
        self._reset_all_red(clear_alloc=True)   # clear everything on full stop
        self.status_var.set("⏸  Simulation stopped.  Press  ▶  to restart.")
        self._log("  Simulation stopped.")

    # ─────────── Signal Cycle ─────────────────────────────────────────────

    def _run_lane(self):
        if not self.running:
            return

        self.current_lane = self.current_lane % 4
        i = self.current_lane
        cam, dr = CAMERAS[i]

        # ── Read the LIVE count right NOW (at this lane's turn) ─────────
        live_now = self.live_counts[i]

        # Calculate allocated time from the live count at this exact moment
        allocated = float(min(TOTAL_CYCLE, max(MIN_GREEN, MIN_GREEN + live_now)))
        self.times[i] = allocated

        # Freeze this lane — count display locked at live_now
        self.frozen_lanes.add(i)
        self.count_labels[i].config(text=str(live_now))

        # Reset signals without wiping alloc time labels
        self._reset_all_red(clear_alloc=False)
        self._set_green(i)

        # Show this lane's alloc time in green (active)
        self.alloc_labels[i].config(text=f"{allocated:.0f} s", fg=GREEN_ON)

        # Other lanes: show their alloc if already calculated, else 'pending'
        for j in range(4):
            if j == i:
                continue
            if self.times[j] > 0:
                self.alloc_labels[j].config(text=f"{self.times[j]:.0f} s", fg=AMBER)
            else:
                self.alloc_labels[j].config(text="pending…", fg=GREY)

        self.remaining = int(round(allocated))

        # Log this lane's allocation
        self._log(f"  🟢 {cam} ({dr}): count={live_now}  →  {allocated:.0f} s")
        print(f"  {cam} ({dr}): count = {live_now:>4}  →  {allocated:.0f} s")

        self.status_var.set(
            f"🟢  {cam} ({dr})  —  Count: {live_now}  →  {allocated:.0f} s allocated  |  Counting down …"
        )
        self._tick()

    def _tick(self):
        """Decrement timer by 1 s each second, then advance to the next lane."""
        if not self.running:
            return

        i = self.current_lane
        self.timer_labels[i].config(text=f"{self.remaining}s")

        if self.remaining > 0:
            self.remaining -= 1
            self._tick_job = self.root.after(1000, self._tick)
        else:
            # ── Lane finished ────────────────────────────────────────────
            cam, dr = CAMERAS[i]
            self._set_red(i)
            self.timer_labels[i].config(text="──", fg=GREY)

            # ── Reset count: vehicles have cleared, fresh count begins ───
            # 1. Signal the tracker thread to re-baseline its internal counter
            #    so the published count restarts cleanly from 0.
            if COUNT_STORE_AVAILABLE:
                count_store.request_reset(dr)
            # 2. Zero our local copy so _poll_counts doesn't flicker
            self.live_counts[i] = 0
            # 3. Update the count display to 0
            self.count_labels[i].config(text="0")

            # Keep alloc time visible in amber to show what was used
            self.alloc_labels[i].config(text=f"{self.times[i]:.0f} s", fg=AMBER)

            # Unfreeze this lane so _poll_counts can resume live updates
            self.frozen_lanes.discard(i)

            self._log(f"  ✓ {cam} ({dr}) done — count reset to 0, fresh cycle starts")
            self.current_lane += 1
            self._run_lane()

    # ─────────── Cleanup ──────────────────────────────────────────────────

    def on_close(self):
        self.running = False
        if self._tick_job:
            self.root.after_cancel(self._tick_job)
        if self._poll_job:
            self.root.after_cancel(self._poll_job)
        self.root.destroy()


# ─────────────── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()