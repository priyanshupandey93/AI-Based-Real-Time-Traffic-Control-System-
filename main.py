"""
main.py
──────────────────────────────────────────────────────────────
Single entry-point for the AI Adaptive Traffic Signal System.

Launches two windows simultaneously:
  1. main_dashboard.py        –  4-camera vehicle detection feed
                                 (writes live counts to count_store)
  2. simulation_dashboard.py  –  live counts + adaptive signal timing
                                 (reads from count_store every 500 ms)

Usage:
    python main.py
"""

import subprocess
import sys
import os

# ── Reset count store before children start so old values don't bleed in ──
try:
    import count_store
    count_store.reset()
    print("[count_store] Counts reset to zero.")
except ImportError:
    print("[WARNING] count_store.py not found – counts won't be shared.")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    py   = sys.executable

    print("\n" + "=" * 58)
    print("   AI ADAPTIVE TRAFFIC SIGNAL CONTROL SYSTEM")
    print("=" * 58)
    print("  ▶  Starting Vehicle Detection Dashboard …")
    print("  ▶  Starting Signal Simulation Dashboard …")
    print("=" * 58 + "\n")
    print("  HOW TO USE:")
    print("  1. Both windows will open automatically.")
    print("  2. Watch the 'Live Count' column fill up as videos play.")
    print("  3. Click  ▶ Start Simulation  in the Simulation window")
    print("     at any time to run the time-allocation algorithm and")
    print("     start the traffic signal countdown cycle.")
    print("=" * 58 + "\n")

    p1 = subprocess.Popen(
        [py, os.path.join(base, "main_dashboard.py")],
        cwd=base
    )
    p2 = subprocess.Popen(
        [py, os.path.join(base, "simulation_dashboard.py")],
        cwd=base
    )

    try:
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        print("\n[main.py] Ctrl+C received — shutting down both dashboards.")
        p1.terminate()
        p2.terminate()


if __name__ == "__main__":
    main()
