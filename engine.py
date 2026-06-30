"""Detection engine, the core loop, controllable by the web server.

The Engine runs as a background asyncio task. It can be:
  - started / stopped
  - enabled / disabled (toggle, stays running but stops acting)
  - asked to calibrate the current channel

It exposes live status for the dashboard.
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

import config
from calibrate import (
    calibrate as run_calibration,
    calibrate_roi,
    compute_hs_histogram,
    load_profile,
    touch_profile,
)
from capture import FrameCapture
from detector import DetectionState, LogoDetector
from intervention import make_controller, PlaybackController, PlaybackMode

logger = logging.getLogger(__name__)


class EngineState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"         # active detection + intervention
    DISABLED = "disabled"       # running but not acting (detection only for UI)
    CALIBRATING = "calibrating"
    ERROR = "error"


@dataclass
class EngineStatus:
    """Snapshot of engine state for the dashboard."""
    engine_state: str = "stopped"
    detection_state: str = "unknown"
    channel: str | None = None
    confidence_edge: float = 0.0
    confidence_template: float = 0.0
    mode: str | None = None          # "live" or "delayed"
    commercial_duration: float = 0.0
    last_update: float = 0.0
    error: str | None = None
    corner_snapshot_b64: str | None = None  # JPEG of the detected corner, base64


class Engine:
    """Controllable detection and intervention engine."""

    def __init__(self):
        self._state = EngineState.STOPPED
        self._task: asyncio.Task | None = None
        self._controller: PlaybackController | None = None
        self._cap: FrameCapture | None = None
        self._detector: LogoDetector | None = None
        self._channel: str | None = None

        self._shadow = False

        # Live telemetry for the dashboard
        self._detection_state = DetectionState.WATCHING
        self._confidence_edge = 0.0
        self._confidence_template = 0.0
        self._mode: PlaybackMode | None = None
        self._commercial_start: float | None = None
        self._corner_snapshot: np.ndarray | None = None
        self._error: str | None = None

        # Time series history for the chart
        self._history: list[dict] = []
        self._events: list[dict] = []
        self._history_max = 1800  # keep ~30 min at 1 sample/sec

        # Threshold tuning: user labeled samples, organized as sessions so we
        # can trim each window's edges (reaction time padding) without losing
        # data near session boundaries.
        self._marking_kind: str | None = None  # "content" | "commercial" | None
        self._content_sessions: list[list[dict]] = []
        self._commercial_sessions: list[list[dict]] = []
        self._active_session: list[dict] | None = None

        # Running sum of content frame color histograms (color/combo profiles
        # only). Stop Marking rebuilds the color reference from this; Clear
        # resets it. Equivalent to keeping every frame, but ~free on memory.
        self._content_hist_sum: np.ndarray | None = None
        self._content_hist_count: int = 0

        # Session stats
        self._stats = {
            "commercials_detected": 0,
            "total_commercial_time": 0.0,
            "skips": 0,
            "mutes": 0,
            "session_start": None,
        }

    @property
    def state(self) -> EngineState:
        return self._state

    def get_status(self) -> EngineStatus:
        """Build a status snapshot for the API."""
        corner_b64 = None
        if self._corner_snapshot is not None:
            _, buf = cv2.imencode(".jpg", self._corner_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 70])
            corner_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        commercial_duration = 0.0
        if self._commercial_start is not None:
            commercial_duration = time.time() - self._commercial_start

        return EngineStatus(
            engine_state=self._state.value,
            detection_state=self._detection_state.value,
            channel=self._channel,
            confidence_edge=round(self._confidence_edge, 3),
            confidence_template=round(self._confidence_template, 3),
            mode=self._mode.value if self._mode else None,
            commercial_duration=round(commercial_duration, 1),
            last_update=time.time(),
            error=self._error,
            corner_snapshot_b64=corner_b64,
        )

    def _set_intervention_mode(self, mode: "PlaybackMode | None"):
        """Set the playback mode and enable break length decay only for LIVE
        (mute) viewing. In skip mode wall clock time doesn't track the
        broadcast break length, so the decay curve doesn't apply."""
        self._mode = mode
        if self._detector is not None:
            self._detector.set_break_decay(mode == PlaybackMode.LIVE)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    async def start(self, channel: str | None = None, shadow: bool = False):
        """Start the engine.

        If shadow=True, runs detection only (no Apple TV connection).
        Channel must be specified in shadow mode.
        """
        if self._state not in (EngineState.STOPPED, EngineState.ERROR):
            logger.warning("Engine already in state %s, ignoring start.", self._state.value)
            return

        self._state = EngineState.STARTING
        self._error = None
        self._shadow = shadow

        try:
            if not shadow:
                # Connect to Apple TV
                self._controller = make_controller()
                await self._controller.connect()

                # Auto detect channel if not specified
                if channel is None:
                    status = await self._controller.get_playback_status()
                    channel = status.channel_slug
                    if not channel:
                        raise RuntimeError("Could not auto detect channel from Apple TV metadata.")
                    logger.info("Auto detected channel: %s", channel)
            else:
                logger.info("Starting in shadow mode (no Apple TV control).")

            self._channel = channel

            # Load profile if a channel is specified
            if channel:
                profile = load_profile(channel)
                touch_profile(channel)
                self._detector = LogoDetector(profile)
            else:
                self._detector = None

            # Open capture device and flush initial frames (capture card needs
            # a moment to sync with the HDMI signal)
            self._cap = FrameCapture()
            self._cap.open()
            logger.info("Flushing initial frames...")
            for _ in range(30):
                self._cap.grab_frame()

            # Launch the detection loop
            self._state = EngineState.RUNNING
            self._stats = {
                "commercials_detected": 0,
                "total_commercial_time": 0.0,
                "skips": 0,
                "mutes": 0,
                "session_start": time.time(),
            }
            self._history = []
            self._events = []
            self._task = asyncio.create_task(self._loop())
            logger.info("Engine started. Channel: %s (shadow=%s)", channel, shadow)

            # Remember this session so we can auto resume after a restart.
            # Only persist real (non shadow) channel runs.
            if not shadow and channel:
                self._save_last_session(channel)

        except Exception as e:
            self._state = EngineState.ERROR
            self._error = str(e)
            logger.error("Engine failed to start: %s", e)
            await self._cleanup()
            raise

    async def stop(self):
        """Stop the engine and clean up."""
        logger.info("Stopping engine...")
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._cleanup()
        self._state = EngineState.STOPPED
        self._channel = None
        self._detector = None
        self._error = None
        # An explicit Stop means "don't come back up running"
        self._clear_last_session()
        logger.info("Engine stopped.")

    # ------------------------------------------------------------------
    # Session persistence, auto resume across restarts
    # ------------------------------------------------------------------

    def _save_last_session(self, channel: str):
        """Record the actively running channel so we can resume after a restart."""
        try:
            with open(config.LAST_SESSION_FILE, "w") as f:
                json.dump({"channel": channel}, f)
        except OSError as e:
            logger.warning("Could not persist last session: %s", e)

    def _clear_last_session(self):
        """Forget the persisted session (called on explicit Stop)."""
        try:
            os.remove(config.LAST_SESSION_FILE)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Could not clear last session: %s", e)

    async def resume_last_session(self):
        """If a session was running before the last shutdown, start it again.

        Called once on server startup. Failures (capture not ready, Apple TV
        unreachable) are logged and leave the engine STOPPED/ERROR rather than
        crashing startup, the user can retry from the dashboard.
        """
        try:
            with open(config.LAST_SESSION_FILE) as f:
                channel = json.load(f).get("channel")
        except FileNotFoundError:
            return
        except (OSError, ValueError) as e:
            logger.warning("Could not read last session: %s", e)
            return

        if not channel:
            return

        logger.info("Auto resuming last session on channel: %s", channel)
        try:
            await self.start(channel=channel)
        except Exception as e:
            logger.warning("Auto resume failed (start from the dashboard): %s", e)

    def toggle(self) -> bool:
        """Toggle between RUNNING and DISABLED. Returns new enabled state."""
        if self._state == EngineState.RUNNING:
            self._state = EngineState.DISABLED
            logger.info("Engine disabled (detection continues, intervention paused).")
            return False
        elif self._state == EngineState.DISABLED:
            self._state = EngineState.RUNNING
            logger.info("Engine enabled (intervention resumed).")
            return True
        else:
            logger.warning("Cannot toggle in state %s", self._state.value)
            return self._state == EngineState.RUNNING

    # ------------------------------------------------------------------
    # Threshold tuning, user labeled sample collection
    # ------------------------------------------------------------------

    # Frames trimmed from each end of every marking session to forgive
    # reaction time slack on the start/stop button clicks (~1s at 10fps).
    _MARK_TRIM_FRAMES = 10

    def start_marking(self, kind: str) -> dict:
        """Begin a new marking session labeled as 'content' or 'commercial'."""
        if kind not in ("content", "commercial"):
            raise ValueError(f"Invalid mark kind: {kind}")
        self._marking_kind = kind
        self._active_session = []
        if kind == "content":
            self._content_sessions.append(self._active_session)
        else:
            self._commercial_sessions.append(self._active_session)
        logger.info("Began marking session: %s", kind)
        return self._marking_status()

    def stop_marking(self) -> dict:
        """Close the active marking session. Frames already collected are kept.

        Closing a content session rebuilds the color reference from all marked
        content frames (color/combo profiles only); see _rebuild_color_profile.
        """
        prev = self._marking_kind
        self._marking_kind = None
        self._active_session = None
        logger.info("Stopped marking (was: %s)", prev)
        status = self._marking_status()
        if prev == "content":
            rebuilt = self._rebuild_color_profile()
            if rebuilt is not None:
                status["color_profile"] = rebuilt
        return status

    def _rebuild_color_profile(self) -> dict | None:
        """Rebuild the active channel's color reference from marked content frames.

        Averages every content marking frame's hue/saturation histogram (the running sum)
        and renormalizes it the same way calibrate_color does, then replaces
        both the saved `<channel>_hist.npy` (backing up the old one) and the
        live detector reference so it takes effect immediately. No op for
        logo only profiles or when too few frames have been gathered. Returns a
        small status dict when it actually applies, else None.
        """
        mode = getattr(self._detector, "_mode", None) if self._detector is not None else None
        if mode not in ("color", "combo"):
            return None
        if self._content_hist_sum is None or self._content_hist_count == 0:
            return None
        if self._content_hist_count < config.MIN_COLOR_REBUILD_FRAMES:
            logger.info(
                "Color rebuild skipped: only %d content frames (need %d).",
                self._content_hist_count, config.MIN_COLOR_REBUILD_FRAMES,
            )
            return None

        new_ref = self._content_hist_sum / self._content_hist_count
        cv2.normalize(new_ref, new_ref, 0, 1, cv2.NORM_MINMAX)
        new_ref = new_ref.astype(np.float32)

        channel = self._channel
        hist_path = os.path.join(config.PROFILES_DIR, f"{channel}_hist.npy")
        # Replace is destructive, keep the previous reference for recovery.
        if os.path.exists(hist_path):
            try:
                shutil.copyfile(hist_path, os.path.join(config.PROFILES_DIR, f"{channel}_hist.bak.npy"))
            except OSError as e:
                logger.warning("Could not back up color profile: %s", e)
        np.save(hist_path, new_ref)
        self._detector._ref_hist = new_ref

        logger.info(
            "Color profile rebuilt for '%s' from %d content frames.",
            channel, self._content_hist_count,
        )
        return {"channel": channel, "frames": self._content_hist_count, "applied": True}

    def clear_samples(self):
        """Discard all collected sessions on both buckets."""
        self._content_sessions = []
        self._commercial_sessions = []
        self._marking_kind = None
        self._active_session = None
        self._content_hist_sum = None
        self._content_hist_count = 0
        logger.info("Cleared all labeled samples.")

    @classmethod
    def _trim_session(cls, session: list[dict]) -> list[dict]:
        """Drop reaction time padding from each end without eating short sessions.

        Trim min(_MARK_TRIM_FRAMES, len // 4) from each side, so very short
        windows still contribute their middle frames.
        """
        n = len(session)
        if n < 5:
            return session
        trim = min(cls._MARK_TRIM_FRAMES, n // 4)
        return session[trim:n - trim] if trim > 0 else session

    @classmethod
    def _all_trimmed(cls, sessions: list[list[dict]]) -> list[dict]:
        out = []
        for s in sessions:
            out.extend(cls._trim_session(s))
        return out

    def _marking_status(self) -> dict:
        return {
            "marking": self._marking_kind,
            "content_count": sum(len(s) for s in self._content_sessions),
            "commercial_count": sum(len(s) for s in self._commercial_sessions),
            "content_sessions": len(self._content_sessions),
            "commercial_sessions": len(self._commercial_sessions),
            "trim_per_side": self._MARK_TRIM_FRAMES,
        }

    def suggest_thresholds(self) -> dict:
        """Compute suggested match + recovery + debounce from labeled samples.

        Each marking session is edge trimmed (reaction time padding) before
        analysis. Percentiles are computed across all trimmed samples per
        bucket; the longest run debounce calculation runs per session and
        takes the max so a session boundary can never inflate the result.
        """
        import numpy as np

        content_samples = self._all_trimmed(self._content_sessions)
        commercial_samples = self._all_trimmed(self._commercial_sessions)

        if not content_samples or not commercial_samples:
            return {
                "ok": False,
                "reason": "Need at least one content session and one commercial session with usable frames.",
                **self._marking_status(),
            }

        def percentiles(samples: list[dict], key: str) -> dict:
            values = np.array([s[key] for s in samples], dtype=np.float32)
            return {
                "min": float(values.min()),
                "p5": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "max": float(values.max()),
            }

        def longest_run_in_session(session: list[dict], signal_key: str, predicate) -> int:
            best = 0
            current = 0
            for s in session:
                if predicate(s[signal_key]):
                    current += 1
                    if current > best:
                        best = current
                else:
                    current = 0
            return best

        def longest_run_across_sessions(sessions: list[list[dict]], signal_key: str, predicate) -> int:
            best = 0
            for raw in sessions:
                trimmed = self._trim_session(raw)
                run = longest_run_in_session(trimmed, signal_key, predicate)
                if run > best:
                    best = run
            return best

        def suggest_for_signal(signal_key: str) -> dict:
            content = percentiles(content_samples, signal_key)
            commercial = percentiles(commercial_samples, signal_key)
            if content["max"] < 0.05 and commercial["max"] < 0.05:
                return {"available": False, "content": content, "commercial": commercial}
            high_commercial = commercial["p95"]
            low_content = content["p5"]
            overlap = high_commercial > low_content
            if overlap:
                match = (commercial["median"] + content["median"]) / 2.0
            else:
                match = (high_commercial + low_content) / 2.0
            recovery = max(match + 0.05, min(content["p25"], content["median"]))
            longest_false_dip = longest_run_across_sessions(
                self._content_sessions, signal_key, lambda v: v < match
            )
            longest_false_spike = longest_run_across_sessions(
                self._commercial_sessions, signal_key, lambda v: v >= match
            )
            debounce = min(longest_false_dip + 1, 30)
            recovery_debounce = min(longest_false_spike + 1, 30)
            return {
                "available": True,
                "content": content,
                "commercial": commercial,
                "match": round(match, 3),
                "recovery": round(recovery, 3),
                "overlap": overlap,
                "debounce": debounce,
                "debounce_uncapped": longest_false_dip + 1,
                "recovery_debounce": recovery_debounce,
                "recovery_debounce_uncapped": longest_false_spike + 1,
                "longest_false_dip": longest_false_dip,
                "longest_false_spike": longest_false_spike,
            }

        return {
            "ok": True,
            "content_count": len(content_samples),
            "commercial_count": len(commercial_samples),
            "content_sessions": len(self._content_sessions),
            "commercial_sessions": len(self._commercial_sessions),
            "trim_per_side": self._MARK_TRIM_FRAMES,
            "logo_template": suggest_for_signal("tmpl"),
            "logo_edge": suggest_for_signal("edge"),
            "color": suggest_for_signal("color"),
        }

    async def disarm(self):
        """Emergency restore, unmute, reset to WATCHING, keep detection running."""
        logger.info("DISARM, restoring normal viewing.")
        if self._controller:
            try:
                await self._controller.unmute()
            except Exception as e:
                logger.warning("Disarm unmute failed: %s", e)
        if self._detector:
            self._detector.reset()
        self._detection_state = DetectionState.WATCHING
        self._commercial_start = None
        self._events.append({"t": round(time.time(), 1), "type": "disarm"})
        logger.info("Disarmed. Normal viewing restored.")


    async def calibrate(self, channel: str | None = None) -> str:
        """Run calibration. If channel is None, auto detects.

        Returns the channel name that was calibrated.
        """
        was_running = self._state in (EngineState.RUNNING, EngineState.DISABLED)
        prev_state = self._state

        if was_running:
            # Pause detection during calibration
            self._state = EngineState.CALIBRATING
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            # Release capture device so calibration can open it
            if self._cap:
                self._cap.close()
                self._cap = None

        try:
            self._state = EngineState.CALIBRATING

            # Auto detect channel
            if channel is None and self._controller:
                status = await self._controller.get_playback_status()
                channel = status.channel_slug
            if channel is None:
                # Try connecting if we don't have a connection
                atv = make_controller()
                await atv.connect()
                status = await atv.get_playback_status()
                channel = status.channel_slug
                await atv.disconnect()

            if not channel:
                raise RuntimeError("Could not determine channel for calibration.")

            logger.info("Starting calibration for channel: %s", channel)

            # Run calibration (blocking I/O, run in executor to not block event loop)
            await asyncio.get_running_loop().run_in_executor(
                None, run_calibration, channel
            )

            logger.info("Calibration complete for channel: %s", channel)

            # Reload profile and restart detection if we were running
            if was_running:
                self._channel = channel
                profile = load_profile(channel)
                touch_profile(channel)
                self._detector = LogoDetector(profile)
                # Reopen capture device
                self._cap = FrameCapture()
                self._cap.open()
                self._state = prev_state
                self._task = asyncio.create_task(self._loop())
                logger.info("Detection restarted with new profile.")
            else:
                self._state = EngineState.STOPPED

            return channel

        except Exception as e:
            self._state = EngineState.ERROR
            self._error = str(e)
            logger.error("Calibration failed: %s", e)
            if was_running:
                await self._cleanup()
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _cleanup(self):
        """Release resources."""
        if self._cap:
            self._cap.close()
            self._cap = None
        if self._controller:
            await self._controller.disconnect()
            self._controller = None

    async def _loop(self):
        """Main detection and intervention loop."""
        commercial_mode: PlaybackMode | None = None

        try:
            logger.info("Detection loop started.")
            while True:
                frame = self._cap.grab_frame()
                if frame is None:
                    logger.warning("Failed to grab frame, retrying...")
                    await asyncio.sleep(1)
                    continue

                # Update telemetry for dashboard (always, even without detector)
                self._update_telemetry(frame)

                # No detector = shadow mode with no profile, just show the feed
                if self._detector is None:
                    await asyncio.sleep(config.FRAME_CHECK_INTERVAL)
                    continue

                prev_state = self._detector.state
                new_state = self._detector.check_frame(frame)
                self._detection_state = new_state

                intervening = self._state == EngineState.RUNNING and not self._shadow

                # -- Transition: WATCHING -> COMMERCIAL --
                if prev_state == DetectionState.WATCHING and new_state == DetectionState.COMMERCIAL:
                    self._commercial_start = time.time()
                    self._stats["commercials_detected"] += 1
                    logger.info("=== COMMERCIAL BREAK DETECTED ===")
                    self._events.append({"t": round(time.time(), 1), "type": "commercial_start"})

                    if intervening:
                        # Always mute first
                        logger.info("Muting.")
                        await self._controller.mute()
                        self._stats["mutes"] += 1
                        self._events.append({"t": round(time.time(), 1), "type": "mute"})

                        if config.PLAYBACK_MODE_OVERRIDE:
                            commercial_mode = PlaybackMode(config.PLAYBACK_MODE_OVERRIDE)
                            logger.info("Mode override: %s", commercial_mode.value)
                        else:
                            status = await self._controller.get_playback_status()
                            commercial_mode = status.mode
                        self._set_intervention_mode(commercial_mode)

                        if commercial_mode == PlaybackMode.DELAYED:
                            logger.info("Mode: DELAYED, also skipping.")
                            self._stats["skips"] += 1
                            self._events.append({"t": round(time.time(), 1), "type": "skip"})
                            await self._skip_loop()
                            # Unmute after skip loop, the show should be back
                            logger.info("Skip loop done, unmuting.")
                            await self._controller.unmute()
                            self._events.append({"t": round(time.time(), 1), "type": "unmute"})
                            self._commercial_start = None
                            commercial_mode = None
                        else:
                            logger.info("Mode: LIVE, muted, waiting for show to resume.")

                # -- Transition: COMMERCIAL -> WATCHING --
                elif prev_state == DetectionState.COMMERCIAL and new_state == DetectionState.WATCHING:
                    duration = (
                        time.time() - self._commercial_start
                        if self._commercial_start
                        else 0
                    )
                    logger.info("=== SHOW RESUMED === (commercial lasted %.1fs)", duration)
                    self._stats["total_commercial_time"] += duration
                    self._events.append({"t": round(time.time(), 1), "type": "commercial_end"})

                    if intervening:
                        logger.info("Unmuting.")
                        await self._controller.unmute()
                        self._events.append({"t": round(time.time(), 1), "type": "unmute"})

                    self._set_intervention_mode(None)
                    self._commercial_start = None
                    commercial_mode = None

                # -- Safety timeout --
                if (
                    new_state == DetectionState.COMMERCIAL
                    and self._commercial_start
                    and time.time() - self._commercial_start > config.MAX_COMMERCIAL_DURATION
                ):
                    logger.warning(
                        "Commercial exceeded %ds, possible detection failure. Resetting.",
                        config.MAX_COMMERCIAL_DURATION,
                    )
                    self._detector.reset()
                    self._detection_state = DetectionState.WATCHING

                    if intervening and commercial_mode == PlaybackMode.LIVE:
                        await self._controller.unmute()

                    self._set_intervention_mode(None)
                    self._commercial_start = None
                    commercial_mode = None

                await asyncio.sleep(config.FRAME_CHECK_INTERVAL)

        except asyncio.CancelledError:
            logger.info("Detection loop cancelled.")
            raise
        except Exception as e:
            self._state = EngineState.ERROR
            self._error = str(e)
            logger.error("Detection loop error: %s", e)

    async def _skip_loop(self):
        """Skip through a commercial break in delayed mode.

        Uses skip_forward first. If position doesn't move (e.g. sponsored
        overlay), falls back to seek_forward (right + select).
        """
        max_iterations = int(config.MAX_COMMERCIAL_DURATION / config.SKIP_SETTLE_TIME)
        use_seek = False

        logger.info("Entering skip loop...")

        # First big skip, give the stream time to buffer before checking position
        status_before = await self._controller.get_playback_status()
        await self._controller.skip_forward(config.FIRST_SKIP_COUNT)
        await asyncio.sleep(3.0)
        status_after = await self._controller.get_playback_status()

        moved = status_after.position - status_before.position
        if moved < 5:
            logger.warning(
                "skip_forward had no effect (moved %.0fs). "
                "Falling back to seek (right + select).", moved
            )
            use_seek = True
            await self._controller.seek_forward(taps=8)
            await asyncio.sleep(config.SEEK_SETTLE_TIME)

        for iteration in range(max_iterations):
            # Bail out if we've caught up to live, further skipping is impossible
            if not config.PLAYBACK_MODE_OVERRIDE:
                live_check = await self._controller.get_playback_status()
                if live_check.mode == PlaybackMode.LIVE:
                    logger.info("Skip loop: caught up to live, exiting skip loop.")
                    return

            # Check multiple frames to satisfy debounce without extra skips
            for _ in range(config.DEBOUNCE_FRAMES + 2):
                frame = self._cap.grab_frame()
                if frame is None:
                    await asyncio.sleep(0.5)
                    continue

                state = self._detector.check_frame(frame)
                self._detection_state = state
                self._update_telemetry(frame)

                if state == DetectionState.WATCHING:
                    logger.info("Skip loop: content detected after %d skip(s).", iteration + 1)
                    return
                await asyncio.sleep(config.FRAME_CHECK_INTERVAL)

            # Try to skip again
            if use_seek:
                await self._controller.seek_forward(taps=4)
                await asyncio.sleep(config.SEEK_SETTLE_TIME)
                continue
            else:
                status_before = await self._controller.get_playback_status()
                await self._controller.skip_forward(config.SUBSEQUENT_SKIP_COUNT)
                await asyncio.sleep(3.0)
                status_after = await self._controller.get_playback_status()

                moved = status_after.position - status_before.position
                if moved < 5:
                    logger.warning("skip_forward stopped working, switching to seek.")
                    use_seek = True
                    await self._controller.seek_forward(taps=4)

            await asyncio.sleep(config.SKIP_SETTLE_TIME)

        logger.warning("Skip loop: hit max iterations (%d), giving up.", max_iterations)

    def _update_telemetry(self, frame: np.ndarray):
        """Update dashboard telemetry from the latest frame."""
        from capture import extract_corner_rois

        if self._detector is None:
            # No profile, just store a scaled down full frame as the snapshot
            scale = 0.25
            h, w = frame.shape[:2]
            self._corner_snapshot = cv2.resize(frame, (int(w * scale), int(h * scale)))
            return

        if self._detector._roi_rect is not None:
            x, y, w, h = self._detector._roi_rect
            self._corner_snapshot = frame[y:y+h, x:x+w]
        elif getattr(self._detector, "_mode", "logo") == "color":
            # Color mode without an explicit ROI uses the full frame; downscale for the snapshot.
            scale = 0.25
            h, w = frame.shape[:2]
            self._corner_snapshot = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            self._corner_snapshot = extract_corner_rois(frame)[self._detector.corner]
        self._confidence_edge, self._confidence_template = self._detector.last_confidences
        color_conf = float(getattr(self._detector, "last_color_confidence", 0.0))

        # If the user is labeling a sample window, append every frame's
        # confidences to the active session. Per frame (not throttled) so
        # we get a richer distribution than the 1/sec history sampling.
        if self._marking_kind is not None and self._active_session is not None:
            # Record the raw per frame signals so suggest_thresholds can derive
            # match/recovery/debounce from the labeled distributions.
            self._active_session.append({
                "edge": round(self._confidence_edge, 3),
                "tmpl": round(self._confidence_template, 3),
                "color": round(color_conf, 3),
            })

        # While marking content on a color/combo profile, fold each frame's hue/saturation
        # histogram into a running sum (over the same region detection samples)
        # so Stop Marking can rebuild the color reference from everything marked.
        if self._marking_kind == "content" and getattr(self._detector, "_mode", None) in ("color", "combo"):
            hist = compute_hs_histogram(self._detector.color_region(frame))
            if self._content_hist_sum is None:
                self._content_hist_sum = hist.copy()
            else:
                self._content_hist_sum += hist
            self._content_hist_count += 1

        # Record to history (throttled to ~1/sec)
        now = time.time()
        if not self._history or now - self._history[-1]["t"] >= 1.0:
            self._history.append({
                "t": round(now, 1),
                "edge": round(self._confidence_edge, 3),
                "tmpl": round(self._confidence_template, 3),
                "color": round(color_conf, 3),
                "state": self._detection_state.value,
            })
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max:]
