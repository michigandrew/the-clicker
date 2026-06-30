"""Automatic calibration, detects the network logo corner with zero user input.

Algorithm:
1. Capture frames over a configurable duration while a show is playing.
2. For each of the four corners, run Canny edge detection on every frame.
3. Average the edge maps: persistent edges (the logo) survive, transient
   edges (moving video content) wash out.
4. The corner with the strongest persistent edge signal contains the logo.
5. Save the reference edge profile and a template image as a channel profile.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np

import config
from capture import FrameCapture, extract_corner_rois, is_sync_artifact_frame

logger = logging.getLogger(__name__)


def _compute_edge_map(gray_roi: np.ndarray) -> np.ndarray:
    """Canny edge detection on a grayscale ROI. Returns binary edge map."""
    return cv2.Canny(gray_roi, config.CANNY_LOW, config.CANNY_HIGH)


def compute_hs_histogram(bgr_image: np.ndarray) -> np.ndarray:
    """HSV hue+saturation histogram of an image. Returns a normalized 2D float32 array."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [config.COLOR_HIST_BINS_H, config.COLOR_HIST_BINS_S],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)


def capture_calibration_frames(
    cap: FrameCapture,
    duration: float | None = None,
    fps: float = config.CALIBRATION_FPS,
) -> list[np.ndarray]:
    """Capture frames at a fixed rate for calibration.

    Flushes the first 30 frames before sampling because USB capture cards
    return a green screen / blank frames briefly while syncing to HDMI; if
    those bleed into the calibration buffer they pollute color histograms
    and edge profiles with a saturated artifact.
    """
    # Resolve at call time so the runtime tunable CALIBRATION_DURATION setting
    # takes effect (a definition time default would freeze the value at import).
    if duration is None:
        duration = config.CALIBRATION_DURATION

    logger.info("Flushing initial frames before calibration sample...")
    for _ in range(30):
        cap.grab_frame()

    interval = 1.0 / fps
    total_frames = int(duration * fps)
    frames = []

    logger.info(
        "Capturing %d calibration frames over %ds (%.1f fps)...",
        total_frames, duration, fps,
    )

    skipped_artifacts = 0
    for i in range(total_frames):
        t0 = time.monotonic()
        frame = cap.grab_frame()
        if frame is None:
            logger.warning("Missed calibration frame %d/%d", i + 1, total_frames)
        elif is_sync_artifact_frame(frame):
            skipped_artifacts += 1
            logger.warning("Skipped sync artifact (green) frame %d/%d", i + 1, total_frames)
        else:
            frames.append(frame)
            logger.debug("Captured calibration frame %d/%d", i + 1, total_frames)
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))

    if skipped_artifacts:
        logger.warning("Dropped %d sync artifact frame(s) during calibration.", skipped_artifacts)

    logger.info("Captured %d frames for calibration.", len(frames))
    return frames


def analyze_corners(
    frames: list[np.ndarray],
    persistence_threshold: float = config.EDGE_PERSISTENCE_THRESHOLD,
) -> dict:
    """Analyze all four corners across captured frames.

    Returns a dict with:
      - best_corner: name of the corner with the strongest logo signal
      - scores: dict of corner_name -> persistence score
      - edge_profiles: dict of corner_name -> averaged edge map (float32)
      - templates: dict of corner_name -> averaged grayscale ROI (uint8)
    """
    corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
    n_frames = len(frames)

    # Accumulate edge maps and grayscale ROIs per corner
    edge_accum = {name: None for name in corner_names}
    gray_accum = {name: None for name in corner_names}

    for frame in frames:
        rois = extract_corner_rois(frame)
        for name, roi in rois.items():
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = _compute_edge_map(gray).astype(np.float32) / 255.0

            if edge_accum[name] is None:
                edge_accum[name] = edges
                gray_accum[name] = gray.astype(np.float32)
            else:
                edge_accum[name] += edges
                gray_accum[name] += gray.astype(np.float32)

    # Compute persistence scores and averaged images
    scores = {}
    edge_profiles = {}
    templates = {}

    for name in corner_names:
        # Average edge map: value at each pixel = fraction of frames with that edge
        avg_edges = edge_accum[name] / n_frames
        # Persistent edges: present in at least `persistence_threshold` fraction of frames
        persistent = (avg_edges >= persistence_threshold).astype(np.float32)
        # Score = total persistent edge pixels (higher = more logo like content)
        score = float(np.sum(persistent))
        scores[name] = score
        edge_profiles[name] = avg_edges
        templates[name] = (gray_accum[name] / n_frames).astype(np.uint8)

        logger.info("Corner %-12s: persistence score = %.1f", name, score)

    best_corner = max(scores, key=scores.get)
    best_score = scores[best_corner]
    second_best = sorted(scores.values(), reverse=True)[1]

    logger.info("Best corner: %s (score=%.1f)", best_corner, best_score)

    # If the best corner barely stands out from the rest, the signal is weak, 
    # likely calibrated during a commercial or content without a logo.
    if best_score < 50:
        logger.warning(
            "Very low persistence score (%.1f). The logo may not be visible, "
            "make sure a show is playing (not a commercial) and try again.",
            best_score,
        )
    elif best_score < second_best * 1.5:
        logger.warning(
            "Best corner score (%.1f) is not much higher than second best (%.1f). "
            "Logo detection may be unreliable. Consider recalibrating.",
            best_score, second_best,
        )

    return {
        "best_corner": best_corner,
        "scores": scores,
        "edge_profiles": edge_profiles,
        "templates": templates,
    }


def save_profile(
    channel_name: str,
    best_corner: str,
    edge_profile: np.ndarray,
    template: np.ndarray,
    scores: dict,
    profiles_dir: str = config.PROFILES_DIR,
    roi_rect: tuple | None = None,
):
    """Save a logo mode channel profile to disk.

    Creates:
      - profiles/<channel>.json, metadata (corner, ROI size, scores)
      - profiles/<channel>_edges.npy, reference edge map (float32)
      - profiles/<channel>_template.png, reference grayscale template

    If roi_rect is provided as (x, y, w, h), the profile stores custom
    ROI coordinates instead of a corner name.
    """
    os.makedirs(profiles_dir, exist_ok=True)

    json_path = os.path.join(profiles_dir, f"{channel_name}.json")
    edges_path = os.path.join(profiles_dir, f"{channel_name}_edges.npy")
    template_path = os.path.join(profiles_dir, f"{channel_name}_template.png")

    # Preserve last_used_at from existing profile if it exists
    last_used = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            try:
                old = json.load(f)
                last_used = old.get("last_used_at")
            except json.JSONDecodeError:
                pass

    metadata = {
        "channel": channel_name,
        "mode": "logo",
        "corner": best_corner,
        "roi_width": config.CORNER_ROI_WIDTH,
        "roi_height": config.CORNER_ROI_HEIGHT,
        "scores": scores,
        "edge_persistence_threshold": config.EDGE_PERSISTENCE_THRESHOLD,
        "logo_match_threshold": config.LOGO_MATCH_THRESHOLD,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if last_used:
        metadata["last_used_at"] = last_used
    if roi_rect is not None:
        metadata["roi_rect"] = list(roi_rect)
        metadata["roi_width"] = roi_rect[2]
        metadata["roi_height"] = roi_rect[3]

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    np.save(edges_path, edge_profile)
    cv2.imwrite(template_path, template)

    logger.info("Profile saved: %s", json_path)
    logger.info("  Edge profile: %s", edges_path)
    logger.info("  Template: %s", template_path)

    return json_path


def save_color_profile(
    channel_name: str,
    histogram: np.ndarray,
    preview_bgr: np.ndarray,
    roi_rect: tuple | None = None,
    profiles_dir: str = config.PROFILES_DIR,
):
    """Save a color mode channel profile.

    Creates:
      - profiles/<channel>.json, metadata (mode=color, threshold)
      - profiles/<channel>_hist.npy, reference HSV hue/saturation histogram
      - profiles/<channel>_template.png, color preview frame (BGR)
    """
    os.makedirs(profiles_dir, exist_ok=True)

    json_path = os.path.join(profiles_dir, f"{channel_name}.json")
    hist_path = os.path.join(profiles_dir, f"{channel_name}_hist.npy")
    template_path = os.path.join(profiles_dir, f"{channel_name}_template.png")

    last_used = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            try:
                last_used = json.load(f).get("last_used_at")
            except json.JSONDecodeError:
                pass

    # Clean up any stale logo mode artifacts so loaders don't pick them up
    edges_path = os.path.join(profiles_dir, f"{channel_name}_edges.npy")
    if os.path.exists(edges_path):
        os.remove(edges_path)

    metadata = {
        "channel": channel_name,
        "mode": "color",
        "color_match_threshold": config.COLOR_MATCH_THRESHOLD,
        "hist_bins": [config.COLOR_HIST_BINS_H, config.COLOR_HIST_BINS_S],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if last_used:
        metadata["last_used_at"] = last_used
    if roi_rect is not None:
        metadata["roi_rect"] = list(roi_rect)
        metadata["roi_width"] = roi_rect[2]
        metadata["roi_height"] = roi_rect[3]

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    np.save(hist_path, histogram)
    cv2.imwrite(template_path, preview_bgr)

    logger.info("Color profile saved: %s", json_path)
    logger.info("  Histogram: %s", hist_path)
    logger.info("  Preview: %s", template_path)

    return json_path


def load_profile(channel_name: str, profiles_dir: str = config.PROFILES_DIR) -> dict:
    """Load a saved channel profile (logo or color mode).

    Returns a dict with mode specific arrays loaded into memory plus all JSON metadata.
    """
    json_path = os.path.join(profiles_dir, f"{channel_name}.json")
    template_path = os.path.join(profiles_dir, f"{channel_name}_template.png")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No profile found for channel '{channel_name}'. "
            f"Run: python calibrate.py {channel_name}"
        )

    with open(json_path) as f:
        metadata = json.load(f)

    mode = metadata.get("mode", "logo")

    if mode == "color":
        hist_path = os.path.join(profiles_dir, f"{channel_name}_hist.npy")
        histogram = np.load(hist_path)
        preview = cv2.imread(template_path, cv2.IMREAD_COLOR) if os.path.exists(template_path) else None
        return {
            **metadata,
            "histogram": histogram,
            "preview": preview,
        }

    if mode == "combo":
        edges_path = os.path.join(profiles_dir, f"{channel_name}_edges.npy")
        hist_path = os.path.join(profiles_dir, f"{channel_name}_hist.npy")
        preview_path = os.path.join(profiles_dir, f"{channel_name}_preview.png")
        edge_profile = np.load(edges_path)
        histogram = np.load(hist_path)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        preview = cv2.imread(preview_path, cv2.IMREAD_COLOR) if os.path.exists(preview_path) else None
        if template is None:
            raise RuntimeError(f"Failed to load template image: {template_path}")
        return {
            **metadata,
            "edge_profile": edge_profile,
            "template": template,
            "histogram": histogram,
            "preview": preview,
        }

    # Logo mode
    edges_path = os.path.join(profiles_dir, f"{channel_name}_edges.npy")
    edge_profile = np.load(edges_path)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise RuntimeError(f"Failed to load template image: {template_path}")

    return {
        **metadata,
        "edge_profile": edge_profile,
        "template": template,
    }


def touch_profile(channel_name: str, profiles_dir: str = config.PROFILES_DIR):
    """Update last_used_at timestamp for a profile."""
    json_path = os.path.join(profiles_dir, f"{channel_name}.json")
    if not os.path.exists(json_path):
        return
    with open(json_path) as f:
        metadata = json.load(f)
    metadata["last_used_at"] = datetime.now(timezone.utc).isoformat()
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)


def delete_profile(channel_name: str, profiles_dir: str = config.PROFILES_DIR):
    """Delete a channel profile and its associated files."""
    for suffix in [".json", "_edges.npy", "_hist.npy", "_template.png", "_preview.png"]:
        path = os.path.join(profiles_dir, f"{channel_name}{suffix}")
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted: %s", path)


def calibrate(
    channel_name: str,
    device=config.CAPTURE_DEVICE,
    duration: float | None = None,
) -> str:
    """Run the full calibration flow. Returns path to saved profile JSON."""
    with FrameCapture(device) as cap:
        frames = capture_calibration_frames(cap, duration=duration)

    if len(frames) < 10:
        raise RuntimeError(
            f"Only captured {len(frames)} frames, need at least 10 for reliable calibration."
        )

    result = analyze_corners(frames)

    path = save_profile(
        channel_name=channel_name,
        best_corner=result["best_corner"],
        edge_profile=result["edge_profiles"][result["best_corner"]],
        template=result["templates"][result["best_corner"]],
        scores=result["scores"],
    )

    return path


def calibrate_roi(
    channel_name: str,
    x: int, y: int, w: int, h: int,
    device=config.CAPTURE_DEVICE,
    duration: float | None = None,
) -> str:
    """Calibrate using a user selected ROI rectangle. Returns path to profile JSON."""
    with FrameCapture(device) as cap:
        frames = capture_calibration_frames(cap, duration=duration)

    if len(frames) < 10:
        raise RuntimeError(
            f"Only captured {len(frames)} frames, need at least 10 for reliable calibration."
        )

    n_frames = len(frames)
    edge_accum = None
    gray_accum = None

    for frame in frames:
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = _compute_edge_map(gray).astype(np.float32) / 255.0

        if edge_accum is None:
            edge_accum = edges
            gray_accum = gray.astype(np.float32)
        else:
            edge_accum += edges
            gray_accum += gray.astype(np.float32)

    avg_edges = edge_accum / n_frames
    persistent = (avg_edges >= config.EDGE_PERSISTENCE_THRESHOLD).astype(np.float32)
    score = float(np.sum(persistent))
    template = (gray_accum / n_frames).astype(np.uint8)

    logger.info("Custom ROI (%d,%d,%d,%d): persistence score = %.1f", x, y, w, h, score)

    path = save_profile(
        channel_name=channel_name,
        best_corner="custom",
        edge_profile=avg_edges,
        template=template,
        scores={"custom": score},
        roi_rect=(x, y, w, h),
    )

    return path


def save_combo_profile(
    channel_name: str,
    logo_rect: tuple,
    edge_profile: np.ndarray,
    template_gray: np.ndarray,
    histogram: np.ndarray,
    preview_bgr: np.ndarray,
    score: float,
    profiles_dir: str = config.PROFILES_DIR,
):
    """Save a combo (logo + color) channel profile.

    Files produced:
      - <channel>.json, metadata (mode=combo, logo_rect, thresholds)
      - <channel>_edges.npy, logo edge map within the logo rect
      - <channel>_template.png, grayscale logo template within the logo rect
      - <channel>_hist.npy, full frame HSV hue/saturation histogram
      - <channel>_preview.png, full frame BGR preview
    """
    os.makedirs(profiles_dir, exist_ok=True)

    json_path = os.path.join(profiles_dir, f"{channel_name}.json")
    edges_path = os.path.join(profiles_dir, f"{channel_name}_edges.npy")
    hist_path = os.path.join(profiles_dir, f"{channel_name}_hist.npy")
    template_path = os.path.join(profiles_dir, f"{channel_name}_template.png")
    preview_path = os.path.join(profiles_dir, f"{channel_name}_preview.png")

    last_used = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            try:
                last_used = json.load(f).get("last_used_at")
            except json.JSONDecodeError:
                pass

    metadata = {
        "channel": channel_name,
        "mode": "combo",
        "corner": "custom",
        "roi_rect": list(logo_rect),
        "roi_width": logo_rect[2],
        "roi_height": logo_rect[3],
        "scores": {"custom": score},
        "edge_persistence_threshold": config.EDGE_PERSISTENCE_THRESHOLD,
        "logo_match_threshold": config.LOGO_MATCH_THRESHOLD,
        "color_match_threshold": config.COLOR_MATCH_THRESHOLD,
        "hist_bins": [config.COLOR_HIST_BINS_H, config.COLOR_HIST_BINS_S],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if last_used:
        metadata["last_used_at"] = last_used

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    np.save(edges_path, edge_profile)
    np.save(hist_path, histogram)
    cv2.imwrite(template_path, template_gray)
    cv2.imwrite(preview_path, preview_bgr)

    logger.info("Combo profile saved: %s", json_path)
    return json_path


def calibrate_combo(
    channel_name: str,
    logo_rect: tuple,
    device=config.CAPTURE_DEVICE,
    duration: float | None = None,
) -> str:
    """Build a combined logo+color profile from a single 30s sample.

    The logo rectangle is used for edge/template matching; the full frame is
    used for the color histogram. Detection treats a frame as content if
    either the logo or the color match passes its threshold.
    """
    x, y, w, h = logo_rect
    with FrameCapture(device) as cap:
        frames = capture_calibration_frames(cap, duration=duration)

    if len(frames) < 10:
        raise RuntimeError(
            f"Only captured {len(frames)} frames, need at least 10 for combo calibration."
        )

    # Logo accumulators (within the rect)
    edge_accum = None
    gray_accum = None
    # Color accumulator (full frame)
    hist_accum = None

    for frame in frames:
        # Logo signals
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = _compute_edge_map(gray).astype(np.float32) / 255.0
        if edge_accum is None:
            edge_accum = edges
            gray_accum = gray.astype(np.float32)
        else:
            edge_accum += edges
            gray_accum += gray.astype(np.float32)

        # Color signal (full frame)
        hist = compute_hs_histogram(frame)
        hist_accum = hist if hist_accum is None else hist_accum + hist

    n = len(frames)
    avg_edges = edge_accum / n
    persistent = (avg_edges >= config.EDGE_PERSISTENCE_THRESHOLD).astype(np.float32)
    score = float(np.sum(persistent))
    template_gray = (gray_accum / n).astype(np.uint8)
    avg_hist = hist_accum / n
    cv2.normalize(avg_hist, avg_hist, 0, 1, cv2.NORM_MINMAX)

    logger.info(
        "Combo profile: %d frames, logo_rect=%s, persistence=%.1f, hist_sum=%.2f",
        n, logo_rect, score, float(avg_hist.sum()),
    )

    return save_combo_profile(
        channel_name=channel_name,
        logo_rect=logo_rect,
        edge_profile=avg_edges,
        template_gray=template_gray,
        histogram=avg_hist.astype(np.float32),
        preview_bgr=frames[-1],
        score=score,
    )


def calibrate_color(
    channel_name: str,
    roi_rect: tuple | None = None,
    device=config.CAPTURE_DEVICE,
    duration: float | None = None,
) -> str:
    """Build a color signature profile by averaging HSV hue/saturation histograms over a sample.

    Captures `duration` seconds of frames, computes a normalized hue/saturation
    histogram of the (optional) ROI in each frame, then averages and renormalizes.
    Detection later compares incoming frame histograms against this reference.
    """
    with FrameCapture(device) as cap:
        frames = capture_calibration_frames(cap, duration=duration)

    if len(frames) < 10:
        raise RuntimeError(
            f"Only captured {len(frames)} frames, need at least 10 for color calibration."
        )

    accum = None
    for frame in frames:
        if roi_rect is not None:
            x, y, w, h = roi_rect
            region = frame[y:y+h, x:x+w]
        else:
            region = frame
        hist = compute_hs_histogram(region)
        accum = hist if accum is None else accum + hist

    avg_hist = accum / len(frames)
    cv2.normalize(avg_hist, avg_hist, 0, 1, cv2.NORM_MINMAX)

    # Use the last frame as the preview (BGR, optionally cropped)
    last = frames[-1]
    preview = last
    if roi_rect is not None:
        x, y, w, h = roi_rect
        preview = last[y:y+h, x:x+w]

    logger.info(
        "Color profile: %d frames, ROI=%s, hist sum=%.2f",
        len(frames), roi_rect, float(avg_hist.sum()),
    )

    return save_color_profile(
        channel_name=channel_name,
        histogram=avg_hist.astype(np.float32),
        preview_bgr=preview,
        roi_rect=roi_rect,
    )


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate a channel's logo profile for ad detection."
    )
    parser.add_argument(
        "channel",
        help="Channel name (e.g., nbc, abc, espn). Used as the profile filename.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=config.CALIBRATION_DURATION,
        help=f"Seconds to capture (default: {config.CALIBRATION_DURATION})",
    )
    parser.add_argument(
        "--device",
        default=config.CAPTURE_DEVICE,
        help=f"Capture device (default: {config.CAPTURE_DEVICE})",
    )
    args = parser.parse_args()

    path = calibrate(args.channel, device=args.device, duration=args.duration)
    print(f"\nCalibration complete. Profile saved to: {path}")
    print("You can now run the detector with this channel profile.")


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )
    main()
