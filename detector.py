"""Logo detection, determines whether the network logo is present in a frame.

Uses a saved channel profile (from calibrate.py) with two detection methods:
1. Edge correlation: compare Canny edges against the persistent edge reference.
2. Template matching: cv2.matchTemplate against the grayscale reference.

The final confidence is the max of both methods. Includes debounce logic to
avoid false triggers from scene transitions or momentary occlusion.
"""

import logging
import time
from enum import Enum

import cv2
import numpy as np

import config
from capture import extract_corner_rois

logger = logging.getLogger(__name__)


def exit_friction(
    t: float,
    strength: float,
    boundaries: list[float],
    window: float,
    max_frames: int,
) -> tuple[int, bool]:
    """Break length decay curve.

    For a commercial of age `t` seconds, return how many EXTRA consecutive
    "game is back" frames to require before exiting (anti blip), plus an
    `ease_threshold` flag.

    - Near a real boundary (within `window`): baseline, no extra frames.
    - Off grid (between boundaries): extra frames ramp up with distance from
      the nearest boundary, peaking at the gap midpoint, capped at
      `max_frames * strength`.
    - Past the longest boundary: the break is almost surely ending, so ease
      (drop hysteresis) and add no friction.

    Pure function, no clock, no globals, so it is fully unit testable.
    """
    if strength <= 0 or not boundaries:
        return (0, False)

    bs = sorted(boundaries)
    if t >= bs[-1]:
        return (0, True)  # past the longest horizon: expect the end

    nearest = min(abs(t - b) for b in bs)
    if nearest <= window:
        return (0, False)  # plausible exit time

    # Saturation distance = half the smallest boundary spacing, so the gap
    # midpoint reaches full friction.
    if len(bs) >= 2:
        spacing = min(b2 - b1 for b1, b2 in zip(bs, bs[1:]))
    else:
        spacing = bs[0]
    sat = max(window + 1.0, spacing / 2.0)

    depth = (nearest - window) / (sat - window)
    depth = max(0.0, min(1.0, depth))
    return (int(round(max_frames * strength * depth)), False)


class DetectionState(Enum):
    WATCHING = "watching"       # show content, logo present
    COMMERCIAL = "commercial"   # commercial break, logo absent


class LogoDetector:
    """Detects show vs commercial state from a calibrated profile.

    Supports two modes:
    - "logo": edge correlation + template matching against a saved logo region
    - "color": HSV hue/saturation histogram correlation against a saved baseline
    """

    def __init__(self, profile: dict):
        self._mode = profile.get("mode", "logo")
        self._roi_rect = profile.get("roi_rect")  # logo rect (or color rect for "color" mode)

        # Debounce state
        self._current_state = DetectionState.WATCHING
        self._candidate_state = DetectionState.WATCHING
        self._consecutive_count = 0
        self._last_edge_conf = 0.0
        self._last_tmpl_conf = 0.0
        self._last_color_conf = 0.0
        # Anti flapping: timestamp of the last COMMERCIAL->WATCHING transition.
        # Reentering within REENTRY_COOLDOWN_SECONDS demands a much longer
        # debounce, so an exit then reenter waffle can't toggle the TV.
        self._last_commercial_exit_t: float | None = None
        # Break length decay: timestamp of the current break's entry, and
        # whether decay is active (engine enables it only in mute/live mode).
        self._commercial_enter_t: float | None = None
        self._break_decay_enabled: bool = False

        if self._mode == "color":
            self._corner = "color"
            self._ref_hist = profile["histogram"]
            logger.info(
                "LogoDetector initialized: mode=color threshold=%.2f roi=%s",
                config.COLOR_MATCH_THRESHOLD, self._roi_rect,
            )
            return

        if self._mode == "combo":
            self._corner = "custom"
            self._edge_ref = profile["edge_profile"]
            self._template = profile["template"]
            self._ref_hist = profile["histogram"]
            self._edge_ref_binary = (
                self._edge_ref >= config.EDGE_PERSISTENCE_THRESHOLD
            ).astype(np.float32)
            logger.info(
                "LogoDetector initialized: mode=combo logo_rect=%s logo_thr=%.2f color_thr=%.2f",
                self._roi_rect, config.LOGO_MATCH_THRESHOLD, config.COLOR_MATCH_THRESHOLD,
            )
            return

        # Logo mode
        self._corner = profile["corner"]
        self._edge_ref = profile["edge_profile"]
        self._template = profile["template"]
        self._edge_ref_binary = (
            self._edge_ref >= config.EDGE_PERSISTENCE_THRESHOLD
        ).astype(np.float32)

        logger.info(
            "LogoDetector initialized: mode=logo corner=%s threshold=%.2f",
            self._corner, config.LOGO_MATCH_THRESHOLD,
        )

    def _required_debounce(self, now: float) -> int:
        """Consecutive frames needed to flip state, evaluated at time `now`.

        Direction aware: entering COMMERCIAL shortly after exiting one demands
        sustained evidence (REENTRY_DEBOUNCE_FRAMES), anti flapping. Baseline
        exit stays fast: staying wrongly muted is the worst failure mode.
        """
        recovering = self._current_state == DetectionState.COMMERCIAL
        if self._mode == "color":
            base = config.COLOR_RECOVERY_DEBOUNCE_FRAMES if recovering else config.COLOR_DEBOUNCE_FRAMES
        else:
            base = config.RECOVERY_DEBOUNCE_FRAMES if recovering else config.DEBOUNCE_FRAMES
        if not recovering and self._in_reentry_cooldown(now):
            base = max(base, config.REENTRY_DEBOUNCE_FRAMES)
        if recovering and self._break_decay_enabled and self._commercial_enter_t is not None:
            extra, _ = exit_friction(
                now - self._commercial_enter_t,
                config.BREAK_DECAY_STRENGTH,
                config.BREAK_BOUNDARIES,
                config.BREAK_BOUNDARY_WINDOW,
                config.EXIT_DECAY_MAX_FRAMES,
            )
            base += extra
        return base

    def _in_reentry_cooldown(self, now: float) -> bool:
        """True while we're within the post exit window of extra entry friction."""
        return (
            self._last_commercial_exit_t is not None
            and (now - self._last_commercial_exit_t) < config.REENTRY_COOLDOWN_SECONDS
        )

    def set_break_decay(self, enabled: bool):
        """Enable/disable break length decay (engine turns it on in mute mode)."""
        self._break_decay_enabled = enabled

    @property
    def state(self) -> DetectionState:
        return self._current_state

    @property
    def corner(self) -> str:
        return self._corner

    @property
    def last_confidences(self) -> tuple[float, float]:
        """(edge_confidence, template_confidence) from the most recent check_frame."""
        return (self._last_edge_conf, self._last_tmpl_conf)

    @property
    def last_color_confidence(self) -> float:
        """Color correlation from the most recent check_frame (combo + color modes)."""
        return self._last_color_conf

    def color_region(self, frame: np.ndarray) -> np.ndarray:
        """The region of the frame that the color signal samples.

        Combo always uses the full frame; color mode uses the ROI rect or full frame.
        """
        if self._mode == "combo":
            return frame
        if self._mode == "color":
            if self._roi_rect is not None:
                x, y, w, h = self._roi_rect
                return frame[y:y+h, x:x+w]
            return frame
        return frame  # logo mode doesn't use color, but return something safe

    def reset(self):
        """Force reset to WATCHING state (e.g. after a safety timeout)."""
        self._current_state = DetectionState.WATCHING
        self._candidate_state = DetectionState.WATCHING
        self._consecutive_count = 0
        self._commercial_enter_t = None
        # A forced reset means detection was misbehaving, apply the same
        # reentry friction so it can't immediately flip back to commercial.
        self._last_commercial_exit_t = time.time()

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract the relevant ROI from a full frame."""
        if self._roi_rect is not None:
            x, y, w, h = self._roi_rect
            return frame[y:y+h, x:x+w]
        if self._mode == "color":
            # Color mode without an explicit ROI uses the full frame.
            return frame
        rois = extract_corner_rois(frame)
        return rois[self._corner]

    def _edge_confidence(self, gray_roi: np.ndarray) -> float:
        """Compute edge based confidence that the logo is present.

        Runs Canny on the current ROI, then computes the overlap with the
        reference persistent edge map.
        """
        edges = cv2.Canny(gray_roi, config.CANNY_LOW, config.CANNY_HIGH).astype(np.float32) / 255.0

        # Overlap: fraction of reference edge pixels that are also edges now
        ref_sum = np.sum(self._edge_ref_binary)
        if ref_sum == 0:
            return 0.0

        overlap = np.sum(edges * self._edge_ref_binary)
        return float(overlap / ref_sum)

    def _template_confidence(self, gray_roi: np.ndarray) -> float:
        """Compute template matching confidence that the logo is present."""
        result = cv2.matchTemplate(
            gray_roi, self._template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max(0.0, max_val))

    def _color_confidence(self, bgr_roi: np.ndarray) -> float:
        """HSV histogram correlation between the current ROI and the saved baseline."""
        from calibrate import compute_hs_histogram
        hist = compute_hs_histogram(bgr_roi)
        return float(cv2.compareHist(self._ref_hist, hist, cv2.HISTCMP_CORREL))

    def _commit_state(self, raw_state: "DetectionState", now: float,
                      confidence: float | None = None) -> "DetectionState":
        """Run debounce + state transition bookkeeping for one frame's raw_state."""
        if raw_state == self._candidate_state:
            self._consecutive_count += 1
        else:
            self._candidate_state = raw_state
            self._consecutive_count = 1

        if (
            self._candidate_state != self._current_state
            and self._consecutive_count >= self._required_debounce(now)
        ):
            old = self._current_state
            self._current_state = self._candidate_state
            if old == DetectionState.WATCHING and self._current_state == DetectionState.COMMERCIAL:
                self._commercial_enter_t = now
            elif old == DetectionState.COMMERCIAL and self._current_state == DetectionState.WATCHING:
                self._last_commercial_exit_t = now
                self._commercial_enter_t = None
            conf_str = "n/a" if confidence is None else "%.3f" % confidence
            logger.info(
                "STATE CHANGE: %s -> %s (after %d consecutive frames, confidence=%s)",
                old.value, self._current_state.value, self._consecutive_count, conf_str,
            )

        return self._current_state

    def check_frame(self, frame: np.ndarray, now: float | None = None) -> DetectionState:
        """Analyze a single frame and return the debounced detection state.

        This should be called once per FRAME_CHECK_INTERVAL.
        """
        if now is None:
            now = time.time()
        # Hysteresis: when currently in COMMERCIAL, require the higher recovery
        # threshold to flip back to WATCHING. Prevents flapping around the match
        # threshold (e.g. ESPN halftime show hovering near 0.55).
        in_commercial = self._current_state == DetectionState.COMMERCIAL
        # Past the longest horizon, break length decay eases the exit threshold
        # (drop hysteresis) so a marginal but real return is caught quickly.
        ease = False
        if in_commercial and self._break_decay_enabled and self._commercial_enter_t is not None:
            _, ease = exit_friction(
                now - self._commercial_enter_t,
                config.BREAK_DECAY_STRENGTH,
                config.BREAK_BOUNDARIES,
                config.BREAK_BOUNDARY_WINDOW,
                config.EXIT_DECAY_MAX_FRAMES,
            )
        use_recovery = in_commercial and not ease
        logo_thr = config.LOGO_RECOVERY_THRESHOLD if use_recovery else config.LOGO_MATCH_THRESHOLD
        color_thr = config.COLOR_RECOVERY_THRESHOLD if use_recovery else config.COLOR_MATCH_THRESHOLD

        if self._mode == "color":
            roi = self._extract_roi(frame)
            color_conf = self._color_confidence(roi)
            self._last_edge_conf = 0.0
            self._last_tmpl_conf = color_conf
            self._last_color_conf = color_conf
            confidence = color_conf
            raw_state = (
                DetectionState.WATCHING if color_conf >= color_thr
                else DetectionState.COMMERCIAL
            )
            logger.debug(
                "Detection (color): correlation=%.3f threshold=%.2f (in_commercial=%s)",
                color_conf, color_thr, in_commercial,
            )
        elif self._mode == "combo":
            # Logo signal in the rect, color signal across the full frame.
            x, y, w, h = self._roi_rect
            logo_roi = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(logo_roi, cv2.COLOR_BGR2GRAY)
            edge_conf = self._edge_confidence(gray)
            tmpl_conf = self._template_confidence(gray)
            color_conf = self._color_confidence(frame)
            self._last_edge_conf = edge_conf
            self._last_tmpl_conf = tmpl_conf
            self._last_color_conf = color_conf
            logo_conf = max(edge_conf, tmpl_conf)
            logo_pass = logo_conf >= logo_thr
            color_pass = color_conf >= color_thr
            confidence = max(logo_conf, color_conf)
            raw_state = (
                DetectionState.WATCHING if (logo_pass or color_pass)
                else DetectionState.COMMERCIAL
            )
            logger.debug(
                "Detection (combo): logo=%.3f (>=%.2f? %s) color=%.3f (>=%.2f? %s) in_commercial=%s -> %s",
                logo_conf, logo_thr, logo_pass,
                color_conf, color_thr, color_pass,
                in_commercial, raw_state.value,
            )
        else:
            roi = self._extract_roi(frame)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edge_conf = self._edge_confidence(gray)
            tmpl_conf = self._template_confidence(gray)
            self._last_edge_conf = edge_conf
            self._last_tmpl_conf = tmpl_conf
            self._last_color_conf = 0.0
            confidence = max(edge_conf, tmpl_conf)
            raw_state = (
                DetectionState.WATCHING if confidence >= logo_thr
                else DetectionState.COMMERCIAL
            )
            logger.debug(
                "Detection (logo): edge=%.3f template=%.3f combined=%.3f threshold=%.2f (in_commercial=%s)",
                edge_conf, tmpl_conf, confidence, logo_thr, in_commercial,
            )

        return self._commit_state(raw_state, now, confidence)
