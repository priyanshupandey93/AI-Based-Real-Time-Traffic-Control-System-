"""
count_store.py
──────────────────────────────────────────────────────────────
Shared inter-process vehicle count bridge.

  WRITER : main_dashboard.py  (updates on count change)
  READER : simulation_dashboard.py (polls every 200 ms)

Reliability fixes:
  • Atomic writes via temp-file + os.replace() — the reader
    never sees a half-written file.
  • Read retry loop (3 attempts, 10 ms apart) — survives the
    tiny window during the rename on slow disks.
  • Only writes when the value actually changes — drastically
    reduces file I/O contention.

Cross-process reset mechanism:
  • Both processes are separate — in-memory flags don't work.
  • A dedicated file (.reset_signals.json) is used as the signal.
  • simulation_dashboard calls request_reset(direction) → writes
    True into .reset_signals.json and zeroes the count file.
  • VideoStreamThread calls consume_reset_flag(direction) before
    each push → reads the signal file; if True, clears it and
    recalibrates its internal baseline so counts restart from 0.
"""

import json
import os
import time

# ── File paths ────────────────────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
STORE_FILE   = os.path.join(_DIR, ".traffic_counts.json")
_TMP_FILE    = os.path.join(_DIR, ".traffic_counts.tmp")
_RESET_FILE  = os.path.join(_DIR, ".reset_signals.json")
_RESET_TMP   = os.path.join(_DIR, ".reset_signals.tmp")

DIRECTIONS   = ["North", "South", "West", "East"]
_default     = {d: 0 for d in DIRECTIONS}
_reset_def   = {d: False for d in DIRECTIONS}

# ── Public API ────────────────────────────────────────────────────────────

def reset() -> None:
    """Set all counts and all reset signals to zero/false.
    Called once at startup from main.py.
    """
    _atomic_write(dict(_default), STORE_FILE, _TMP_FILE)
    _atomic_write(dict(_reset_def), _RESET_FILE, _RESET_TMP)


def request_reset(direction: str) -> None:
    """Signal the tracker thread to restart its count from 0.

    Called by simulation_dashboard when a lane's green phase ends.
    Writes a True flag into .reset_signals.json (cross-process safe)
    AND immediately zeroes the lane's count so the simulation reads 0.
    """
    # 1. Set the reset signal flag in the signal file
    signals = _safe_read(_RESET_FILE, _reset_def)
    signals[direction] = True
    _atomic_write(signals, _RESET_FILE, _RESET_TMP)

    # 2. Immediately zero the count in the count file
    data = _safe_read(STORE_FILE, _default)
    data[direction] = 0
    _atomic_write(data, STORE_FILE, _TMP_FILE)


def consume_reset_flag(direction: str) -> bool:
    """Return True (and clear the flag) if a reset was requested.

    Called by VideoStreamThread BEFORE each push.
    When True is returned the thread must recalibrate its internal
    baseline so subsequent pushes count from 0.
    Uses file I/O so it works correctly across separate processes.
    """
    signals = _safe_read(_RESET_FILE, _reset_def)
    if signals.get(direction, False):
        signals[direction] = False
        _atomic_write(signals, _RESET_FILE, _RESET_TMP)
        return True
    return False


def update_count(direction: str, value: int) -> None:
    """
    Overwrite the count for one direction — but only if the value
    actually changed.  This prevents unnecessary disk I/O.
    """
    value = max(0, int(value))
    data = _safe_read(STORE_FILE, _default)
    if data.get(direction) == value:
        return             # nothing changed — skip the write
    data[direction] = value
    _atomic_write(data, STORE_FILE, _TMP_FILE)


def get_all() -> dict:
    """Return a snapshot of all four current counts."""
    return _safe_read(STORE_FILE, _default)


# ── Internal helpers ──────────────────────────────────────────────────────

def _safe_read(filepath: str, default: dict) -> dict:
    """
    Read a JSON file with up to 3 retries (10 ms apart).
    Handles the tiny window during an atomic rename on slow disks.
    """
    for attempt in range(3):
        try:
            with open(filepath, "r") as fh:
                data = json.load(fh)
            # ensure all expected keys exist
            for k, v in default.items():
                data.setdefault(k, v)
            return data
        except Exception:
            if attempt < 2:
                time.sleep(0.01)
    return dict(default)


def _atomic_write(data: dict, filepath: str, tmp_path: str) -> None:
    """
    Write data to a temp file then atomically rename it over filepath.
    Guarantees the reader never sees a partial / corrupt JSON file.
    """
    try:
        with open(tmp_path, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp_path, filepath)   # atomic on Windows & POSIX
    except Exception:
        pass
