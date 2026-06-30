"""FastAPI web server, dashboard and API for controlling the ad blocker."""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import asdict
from functools import partial

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
from calibrate import delete_profile
from capture import FrameCapture
from engine import Engine

logger = logging.getLogger(__name__)

app = FastAPI(title="The Clicker")
engine = Engine()


@app.on_event("startup")
async def _resume_last_session():
    """Auto resume the last running session after a service restart / reboot."""
    await engine.resume_last_session()


# ------------------------------------------------------------------
# API models
# ------------------------------------------------------------------

class StartRequest(BaseModel):
    channel: str | None = None
    shadow: bool = False


class CalibrateRequest(BaseModel):
    channel: str | None = None


class InteractiveCalibrateRequest(BaseModel):
    channel: str
    x: int
    y: int
    w: int
    h: int
    # Seconds of footage to sample. Omit to use the CALIBRATION_DURATION setting.
    duration: int | None = None


class ColorCalibrateRequest(BaseModel):
    channel: str
    # Optional ROI rectangle. If absent, the whole frame is used.
    x: int | None = None
    y: int | None = None
    w: int | None = None
    h: int | None = None
    duration: int | None = None


class ComboCalibrateRequest(BaseModel):
    """Combo profile: logo rect (required) + full frame color signal."""
    channel: str
    x: int
    y: int
    w: int
    h: int
    duration: int | None = None


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Current engine status, polled by the dashboard."""
    return asdict(engine.get_status())


@app.post("/api/start")
async def start_engine(req: StartRequest = StartRequest()):
    """Start the detection engine."""
    try:
        await engine.start(channel=req.channel, shadow=req.shadow)
        return {"ok": True, "state": engine.state.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_engine():
    """Stop the detection engine."""
    await engine.stop()
    return {"ok": True, "state": engine.state.value}


@app.post("/api/disarm")
async def disarm_engine():
    """Emergency restore, unmute, reset detection, keep running."""
    await engine.disarm()
    return {"ok": True}


# ------------------------------------------------------------------
# Threshold tuning, labeled sample collection
# ------------------------------------------------------------------

class MarkRequest(BaseModel):
    kind: str  # "content" | "commercial"


@app.post("/api/mark/start")
async def mark_start(req: MarkRequest):
    """Begin collecting confidence samples labeled as content or commercial."""
    try:
        return engine.start_marking(req.kind)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mark/stop")
async def mark_stop():
    """Stop the active marking window. Samples gathered so far are retained."""
    return engine.stop_marking()


@app.post("/api/mark/clear")
async def mark_clear():
    """Discard all collected samples and reset both buckets."""
    engine.clear_samples()
    return {"ok": True}


@app.get("/api/mark/status")
async def mark_status():
    """Current marking state and sample counts."""
    return engine._marking_status()


@app.get("/api/suggest-thresholds")
async def suggest_thresholds():
    """Compute suggested match + recovery thresholds from labeled samples."""
    return engine.suggest_thresholds()


@app.get("/api/stats")
async def get_stats():
    """Session statistics."""
    stats = dict(engine._stats)
    if stats.get("session_start"):
        stats["session_duration"] = round(time.time() - stats["session_start"], 1)
        stats["time_saved_min"] = round(stats["total_commercial_time"] / 60, 1)
    else:
        stats["session_duration"] = 0
        stats["time_saved_min"] = 0
    return stats


@app.post("/api/toggle")
async def toggle_engine():
    """Toggle between active (intervening) and disabled (monitoring only)."""
    enabled = engine.toggle()
    return {"ok": True, "enabled": enabled, "state": engine.state.value}


@app.post("/api/calibrate")
async def calibrate_channel(req: CalibrateRequest = CalibrateRequest()):
    """Run calibration for a channel. Auto detects if channel is omitted."""
    try:
        channel = await engine.calibrate(channel=req.channel)
        return {"ok": True, "channel": channel}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profiles")
async def list_profiles():
    """List all saved channel profiles."""
    profiles_dir = config.PROFILES_DIR
    if not os.path.exists(profiles_dir):
        return {"profiles": []}

    profiles = []
    for f in os.listdir(profiles_dir):
        if f.endswith(".json"):
            profiles.append(f.removesuffix(".json"))
    return {"profiles": sorted(profiles)}


@app.delete("/api/profile/{channel}")
async def delete_profile_endpoint(channel: str):
    """Delete a channel profile."""
    json_path = os.path.join(config.PROFILES_DIR, f"{channel}.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail=f"No profile for '{channel}'")
    delete_profile(channel)
    return {"ok": True, "deleted": channel}


def _render_palette(histogram: np.ndarray, width: int = 360, height: int = 60, n_colors: int = 12) -> np.ndarray:
    """Render an HSV hue/saturation histogram as a horizontal swatch strip of dominant colors.

    Picks the top `n_colors` bins by weight and lays them out left to right,
    each rectangle's width proportional to its weight. Bin centers are converted
    to BGR with V=255 for visibility.
    """
    hist = histogram.astype(np.float32)
    if hist.sum() <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    h_bins, s_bins = hist.shape
    flat = hist.flatten()
    top_idx = np.argsort(flat)[::-1][:n_colors]
    weights = flat[top_idx]
    weights = weights / weights.sum()

    img = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for idx, w in zip(top_idx, weights):
        hb, sb = divmod(int(idx), s_bins)
        # Bin centers, mapped back to OpenCV HSV ranges (H: 0-180, S: 0-255)
        hue = int((hb + 0.5) * (180.0 / h_bins))
        sat = int((sb + 0.5) * (256.0 / s_bins))
        swatch_hsv = np.array([[[hue, sat, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        ww = max(1, int(round(w * width)))
        img[:, x:x + ww] = bgr
        x += ww
        if x >= width:
            break
    if x < width:
        img[:, x:] = img[:, x - 1:x] if x > 0 else 0
    return img


def _palette_b64(histogram: np.ndarray) -> str:
    """JPEG encode a histogram palette swatch as base64."""
    img = _render_palette(histogram)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/api/profile/{channel}")
async def get_profile_details(channel: str):
    """Return calibration diagnostics for a channel profile.

    Includes: metadata, per corner scores, reference template image, and
    either the reference edge map (logo mode) or the color palette swatch
    (color mode).
    """
    json_path = os.path.join(config.PROFILES_DIR, f"{channel}.json")
    edges_path = os.path.join(config.PROFILES_DIR, f"{channel}_edges.npy")
    hist_path = os.path.join(config.PROFILES_DIR, f"{channel}_hist.npy")
    template_path = os.path.join(config.PROFILES_DIR, f"{channel}_template.png")
    preview_path = os.path.join(config.PROFILES_DIR, f"{channel}_preview.png")

    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail=f"No profile for '{channel}'")

    with open(json_path) as f:
        metadata = json.load(f)

    mode = metadata.get("mode", "logo")

    # Reference template (grayscale for logo + combo, BGR for color)
    template_b64 = None
    if os.path.exists(template_path):
        flag = cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE
        template = cv2.imread(template_path, flag)
        if template is not None:
            _, buf = cv2.imencode(".jpg", template, [cv2.IMWRITE_JPEG_QUALITY, 85])
            template_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    edges_b64 = None
    palette_b64 = None
    if os.path.exists(edges_path) and mode in ("logo", "combo"):
        edge_profile = np.load(edges_path)
        edge_vis = (edge_profile * 255).clip(0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", edge_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        edges_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    if os.path.exists(hist_path) and mode in ("color", "combo"):
        palette_b64 = _palette_b64(np.load(hist_path))

    return {
        **metadata,
        "template_b64": template_b64,
        "edges_b64": edges_b64,
        "palette_b64": palette_b64,
    }


@app.get("/api/live-palette")
async def get_live_palette():
    """Return a swatch of the current frame's color palette (in the active profile's ROI).

    Used by the dashboard to show the live color palette next to the saved one.
    Falls back to the full frame if no profile/ROI is active.
    """
    from calibrate import compute_hs_histogram

    # Borrow the engine's open capture if it's running; otherwise open briefly.
    cap = None
    try:
        if engine._cap is not None:
            frame = engine._cap.grab_frame()
        else:
            cap = FrameCapture()
            cap.open()
            # Capture cards return green sync artifact frames briefly after open.
            for _ in range(30):
                cap.grab_frame()
            frame = cap.grab_frame()

        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to grab frame")

        # If the frame is still a sync artifact (rare midstream glitch), skip it.
        from capture import is_sync_artifact_frame
        if is_sync_artifact_frame(frame):
            raise HTTPException(status_code=503, detail="Capture card returned a sync artifact frame; try again")

        # Determine the region the active profile uses for color (combo always
        # uses the full frame; color may use a rect; logo profiles fall back to
        # the full frame for visualization).
        det = engine._detector
        mode = getattr(det, "_mode", None)
        if det is not None and hasattr(det, "color_region"):
            region = det.color_region(frame)
        else:
            region = frame
        roi_rect = list(det._roi_rect) if (det is not None and det._roi_rect is not None and mode == "color") else None

        hist = compute_hs_histogram(region)
        match = None
        if det is not None and mode in ("color", "combo") and hasattr(det, "_ref_hist"):
            match = float(cv2.compareHist(det._ref_hist, hist, cv2.HISTCMP_CORREL))

        return {
            "palette_b64": _palette_b64(hist),
            "roi_rect": roi_rect,
            "match": match,
            "threshold": config.COLOR_MATCH_THRESHOLD,
            "mode": mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        # Surface the real capture error instead of a generic 500.
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cap is not None:
            cap.close()


TUNABLE_SETTINGS = {
    "FIRST_SKIP_COUNT": int,
    "SUBSEQUENT_SKIP_COUNT": int,
    "SKIP_SETTLE_TIME": float,
    "LOGO_MATCH_THRESHOLD": float,
    "LOGO_RECOVERY_THRESHOLD": float,
    "DEBOUNCE_FRAMES": int,
    "RECOVERY_DEBOUNCE_FRAMES": int,
    "COLOR_MATCH_THRESHOLD": float,
    "COLOR_RECOVERY_THRESHOLD": float,
    "COLOR_DEBOUNCE_FRAMES": int,
    "COLOR_RECOVERY_DEBOUNCE_FRAMES": int,
    "MUTE_STEPS": int,
    "UNMUTE_STEPS": int,
    "PLAYBACK_MODE_OVERRIDE": str,
    "MAX_COMMERCIAL_DURATION": int,
    "REENTRY_COOLDOWN_SECONDS": int,
    "REENTRY_DEBOUNCE_FRAMES": int,
    "CALIBRATION_DURATION": int,
    "BREAK_DECAY_STRENGTH": float,
    "EXIT_DECAY_MAX_FRAMES": int,
}

# Named detection presets, bundles applied through _apply_settings. Each is a
# point on one axis: how eager the system is to call something a commercial.
# Conservative risks a little ad bleed; Aggressive risks clipping a bit of game.
PRESETS = {
    "conservative": {
        # Lowest match thresholds = reluctant to call a commercial (won't mute
        # the game on a brief logo dip); lower recovery = quick to return.
        "DEBOUNCE_FRAMES": 5, "LOGO_MATCH_THRESHOLD": 0.30,
        "LOGO_RECOVERY_THRESHOLD": 0.45, "RECOVERY_DEBOUNCE_FRAMES": 1,
        "COLOR_MATCH_THRESHOLD": 0.50, "COLOR_RECOVERY_THRESHOLD": 0.65,
        "COLOR_DEBOUNCE_FRAMES": 10, "REENTRY_COOLDOWN_SECONDS": 45,
        "BREAK_DECAY_STRENGTH": 0.0,
    },
    "balanced": {
        "DEBOUNCE_FRAMES": 3, "LOGO_MATCH_THRESHOLD": 0.35,
        "LOGO_RECOVERY_THRESHOLD": 0.50, "RECOVERY_DEBOUNCE_FRAMES": 1,
        "COLOR_MATCH_THRESHOLD": 0.55, "COLOR_RECOVERY_THRESHOLD": 0.70,
        "COLOR_DEBOUNCE_FRAMES": 8, "REENTRY_COOLDOWN_SECONDS": 30,
        "BREAK_DECAY_STRENGTH": 0.5,
    },
    "aggressive": {
        # Highest match thresholds = eager to call a commercial (mutes fast on
        # the first dip); higher recovery + full decay = commits to the break.
        "DEBOUNCE_FRAMES": 2, "LOGO_MATCH_THRESHOLD": 0.45,
        "LOGO_RECOVERY_THRESHOLD": 0.55, "RECOVERY_DEBOUNCE_FRAMES": 1,
        "COLOR_MATCH_THRESHOLD": 0.60, "COLOR_RECOVERY_THRESHOLD": 0.75,
        "COLOR_DEBOUNCE_FRAMES": 6, "REENTRY_COOLDOWN_SECONDS": 20,
        "BREAK_DECAY_STRENGTH": 1.0,
    },
}


class PresetRequest(BaseModel):
    name: str


@app.get("/api/settings")
async def get_settings():
    """Return current tunable settings."""
    return {key: getattr(config, key) for key in TUNABLE_SETTINGS}


def _apply_settings(settings: dict) -> dict:
    """Validate + apply a {key: value} bundle onto config. Raises 400 on an
    unknown key. Returns the full current tunable settings snapshot."""
    for key, value in settings.items():
        if key not in TUNABLE_SETTINGS:
            raise HTTPException(status_code=400, detail=f"Unknown setting: {key}")
        cast = TUNABLE_SETTINGS[key]
        setattr(config, key, cast(value))
        logger.info("Setting %s = %s", key, cast(value))
    return {key: getattr(config, key) for key in TUNABLE_SETTINGS}


@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update tunable settings at runtime."""
    return {"ok": True, **_apply_settings(settings)}


@app.post("/api/preset")
async def apply_preset(req: PresetRequest):
    """Apply a named detection preset (conservative/balanced/aggressive)."""
    bundle = PRESETS.get(req.name.lower())
    if bundle is None:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.name}")
    snapshot = _apply_settings(bundle)
    logger.info("Applied preset: %s", req.name)
    return {"ok": True, "preset": req.name.lower(), **snapshot}


@app.get("/api/history")
async def get_history():
    """Return time series confidence data and events for charting."""
    threshold = config.LOGO_MATCH_THRESHOLD
    mode = "logo"
    if engine._detector is not None:
        mode = getattr(engine._detector, "_mode", "logo")
        if mode == "color":
            threshold = config.COLOR_MATCH_THRESHOLD
    return {
        "history": engine._history,
        "events": engine._events,
        "threshold": threshold,
        "color_threshold": config.COLOR_MATCH_THRESHOLD,
        "logo_threshold": config.LOGO_MATCH_THRESHOLD,
        "mode": mode,
    }


@app.get("/api/capture-check")
async def capture_check():
    """Check if the capture card is connected and working."""
    import platform
    from capture import _find_capture_card_index

    result = {"found": False, "device_name": config.CAPTURE_DEVICE_NAME, "index": None, "frame": False, "resolution": None}

    if platform.system() == "Darwin":
        idx = _find_capture_card_index()
        if idx is not None:
            result["found"] = True
            result["index"] = idx
            # Try grabbing a frame
            cap = None
            try:
                cap = FrameCapture(idx)
                cap.open()
                for _ in range(3):
                    cap.grab_frame()
                frame = cap.grab_frame()
                if frame is not None:
                    result["frame"] = True
                    result["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
            except Exception as e:
                result["error"] = str(e)
            finally:
                if cap:
                    cap.close()
        else:
            # List what we can see
            try:
                from AVFoundation import (
                    AVCaptureDeviceDiscoverySession, AVCaptureDevicePositionUnspecified,
                    AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeExternal,
                    AVMediaTypeVideo,
                )
                session = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                    [AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeExternal],
                    AVMediaTypeVideo, AVCaptureDevicePositionUnspecified,
                )
                result["available_devices"] = [d.localizedName() for d in session.devices()]
            except Exception:
                pass
    else:
        # Linux, just check if device path exists
        import os
        path = config.CAPTURE_DEVICE
        result["found"] = os.path.exists(path) if isinstance(path, str) else True
        result["index"] = path

    return result


@app.get("/api/snapshot")
async def get_snapshot():
    """Grab a single frame from the capture card. Works without the engine running."""
    cap = None
    try:
        if engine._cap is not None:
            frame = engine._cap.grab_frame()
        else:
            cap = FrameCapture()
            cap.open()
            # Capture cards return green sync artifact frames briefly after open.
            for _ in range(30):
                cap.grab_frame()
            frame = cap.grab_frame()

        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to grab frame")

        h, w = frame.shape[:2]
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return {"frame_b64": b64, "width": w, "height": h}

    except HTTPException:
        raise
    except Exception as e:
        # Surface the real capture error (e.g. permission denied) instead of a
        # generic "Internal Server Error".
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cap is not None:
            cap.close()


@app.post("/api/calibrate/interactive")
async def calibrate_interactive(req: InteractiveCalibrateRequest):
    """Calibrate using a user selected ROI region."""
    from calibrate import calibrate_roi

    try:
        path = await asyncio.get_running_loop().run_in_executor(
            None, partial(calibrate_roi, req.channel, req.x, req.y, req.w, req.h,
                          duration=req.duration)
        )
        return {"ok": True, "channel": req.channel, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibrate/color")
async def calibrate_color_endpoint(req: ColorCalibrateRequest):
    """Build a color signature profile by averaging HSV histograms over ~30s of footage.

    If x/y/w/h are provided, samples that region; otherwise uses the full frame.
    """
    from calibrate import calibrate_color

    roi = None
    if req.x is not None and req.y is not None and req.w is not None and req.h is not None:
        roi = (req.x, req.y, req.w, req.h)

    try:
        path = await asyncio.get_running_loop().run_in_executor(
            None, partial(calibrate_color, req.channel, roi, duration=req.duration)
        )
        return {"ok": True, "channel": req.channel, "path": path, "roi": roi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibrate/combo")
async def calibrate_combo_endpoint(req: ComboCalibrateRequest):
    """Build a combo (logo + full frame color) profile.

    The rectangle is the logo region for edge/template matching; the color
    histogram is always sampled from the full frame.
    """
    from calibrate import calibrate_combo

    try:
        path = await asyncio.get_running_loop().run_in_executor(
            None, partial(calibrate_combo, req.channel, (req.x, req.y, req.w, req.h),
                          duration=req.duration)
        )
        return {"ok": True, "channel": req.channel, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frame")
async def get_full_frame():
    """Return the latest full frame with the active ROI corner highlighted.

    Used for the diagnostics view so you can see where on screen the
    detector is looking.
    """
    if engine._cap is None or engine._detector is None:
        raise HTTPException(status_code=409, detail="Engine not running")

    frame = engine._cap.grab_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to grab frame")

    # Draw a rectangle around the active ROI corner
    h, w = frame.shape[:2]
    roi_w, roi_h = config.CORNER_ROI_WIDTH, config.CORNER_ROI_HEIGHT
    corner = engine._detector.corner

    corners = {
        "top_left": (0, 0, roi_w, roi_h),
        "top_right": (w - roi_w, 0, w, roi_h),
        "bottom_left": (0, h - roi_h, roi_w, h),
        "bottom_right": (w - roi_w, h - roi_h, w, h),
    }
    x1, y1, x2, y2 = corners[corner]
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Position text label inside the ROI to avoid clipping off screen
    is_top = corner.startswith("top")
    is_right = corner.endswith("right")
    text_x = x1 + 5 if not is_right else x1 + 5
    text_y = y2 - 10 if is_top else y1 + 20
    cv2.putText(
        annotated, f"ROI: {corner}", (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )

    # Encode as JPEG (scaled down to save bandwidth)
    scale = 0.5
    small = cv2.resize(annotated, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {"frame_b64": b64, "corner": corner}


# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the single page dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path) as f:
        return f.read()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    import uvicorn

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
