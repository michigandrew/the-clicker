"""Configuration for the live TV ad blocker."""

import os
import pathlib
import platform
import logging

# ---------------------------------------------------------------------------
# Apple TV
# ---------------------------------------------------------------------------
# Which playback device the engine controls. Only "appletv" ships today; add
# others by implementing PlaybackController in intervention.py (see README).
CONTROLLER_BACKEND = os.environ.get("CONTROLLER_BACKEND", "appletv")

# CHANGE THIS to your own Apple TV's identifier (used by the "appletv" backend).
# Find it by running `atvremote scan` (from the pyatv package) on the same
# network, it's the "Identifier" field. You can also override it without
# editing this file by setting the APPLE_TV_ID environment variable.
APPLE_TV_ID = os.environ.get("APPLE_TV_ID", "YOUR_APPLE_TV_ID")

# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------
MUTE_STEPS = 10                # volume_down presses to fully mute
UNMUTE_STEPS = 7               # volume_up presses to restore normal level
VOLUME_COMMAND_DELAY = 0.1     # seconds between consecutive volume commands

# ---------------------------------------------------------------------------
# Skip control
# ---------------------------------------------------------------------------
FIRST_SKIP_COUNT = 8           # skip_forward presses on first jump (120s)
SUBSEQUENT_SKIP_COUNT = 1      # skip_forward presses after first (15s)
SKIP_SETTLE_TIME = 0.33        # seconds to wait after skip for frame to settle
SEEK_SETTLE_TIME = 1.0         # seconds to wait after seek (right+select) for frame to settle
SKIP_COMMAND_DELAY = 0.1       # seconds between consecutive skip_forward presses

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
DEBOUNCE_FRAMES = 3            # consecutive frames to confirm WATCHING→COMMERCIAL
RECOVERY_DEBOUNCE_FRAMES = 1  # consecutive frames to confirm COMMERCIAL→WATCHING (faster exit)

# Anti flapping: after exiting a commercial, reentering requires sustained
# evidence for a while. A wrong exit corrects itself a little late (cheap);
# a mute/unmute waffle on the TV is the expensive failure.
REENTRY_COOLDOWN_SECONDS = 30  # window after commercial_end with extra entry friction
REENTRY_DEBOUNCE_FRAMES = 30   # consecutive frames (~3s) to reenter during that window
FRAME_CHECK_INTERVAL = 0.1     # seconds between frame captures in main loop
LOGO_MATCH_THRESHOLD = 0.35    # confidence threshold for logo detection (0-1)
LOGO_RECOVERY_THRESHOLD = 0.50 # higher bar required to LEAVE commercial state (hysteresis)
MAX_COMMERCIAL_DURATION = 300  # seconds, safety timeout if detection seems wrong

# ---------------------------------------------------------------------------
# Color profile detection (alternative to logo template matching)
# ---------------------------------------------------------------------------
COLOR_MATCH_THRESHOLD = 0.55   # HSV histogram correlation: 1.0=identical, lower=different
COLOR_RECOVERY_THRESHOLD = 0.70 # higher bar required to LEAVE commercial state (hysteresis)
COLOR_DEBOUNCE_FRAMES = 8      # consecutive frames to confirm WATCHING→COMMERCIAL
COLOR_RECOVERY_DEBOUNCE_FRAMES = 2  # consecutive frames to confirm COMMERCIAL→WATCHING
COLOR_HIST_BINS_H = 50         # hue bins
COLOR_HIST_BINS_S = 60         # saturation bins
MIN_COLOR_REBUILD_FRAMES = 10  # min marked content frames before rebuilding the color profile

# ---------------------------------------------------------------------------
# Live vs delayed
# ---------------------------------------------------------------------------
LIVE_THRESHOLD = 0.90          # position/total_time above this = "live"
PLAYBACK_MODE_OVERRIDE = ""    # "", "live", or "delayed", empty = auto detect

# ---------------------------------------------------------------------------
# Capture device
# ---------------------------------------------------------------------------
if platform.system() == "Darwin":
    CAPTURE_DEVICE = 1         # macOS: use device index (probe if wrong)
else:
    CAPTURE_DEVICE = "/dev/video0"  # Linux: V4L2 device path

CAPTURE_DEVICE_NAME = "USB3.0 Video"  # AVFoundation name for device lookup on macOS
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CALIBRATION_DURATION = 60      # seconds to capture frames during calibration (runtime tunable)
CALIBRATION_FPS = 2            # frames per second during calibration
CORNER_ROI_WIDTH = 200         # pixels, width of each corner ROI
CORNER_ROI_HEIGHT = 150        # pixels, height of each corner ROI
EDGE_PERSISTENCE_THRESHOLD = 0.3  # fraction of frames an edge must appear in to be "persistent"
CANNY_LOW = 50                 # Canny edge detector low threshold
CANNY_HIGH = 150               # Canny edge detector high threshold

# ---------------------------------------------------------------------------
# Break length decay (mute/live mode), resist believing a break ended at an
# implausible time. Breaks cluster near these lengths; an exit far from any of
# them is more likely a false logo blip than a real return.
# ---------------------------------------------------------------------------
BREAK_BOUNDARIES = [30, 60, 90, 120]  # typical break lengths, seconds
BREAK_BOUNDARY_WINDOW = 10            # +/- seconds around a boundary = "plausible"
EXIT_DECAY_MAX_FRAMES = 25            # cap on added exit debounce (~2.5s @ 10fps)
BREAK_DECAY_STRENGTH = 0.5            # 0=off .. 1=full; set by preset

PROFILES_DIR = str(pathlib.Path(__file__).parent / "profiles")

# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------
# Records the last actively running channel so the engine can auto resume after
# a service restart / reboot. Cleared on an explicit Stop.
LAST_SESSION_FILE = str(pathlib.Path(__file__).parent / ".last_session.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
