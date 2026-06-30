import numpy as np

import config
from detector import LogoDetector, DetectionState


def make_logo_detector():
    """Minimal logo mode detector, no real frames needed for state machine tests."""
    profile = {
        "mode": "logo",
        "corner": "bottom_left",
        "edge_profile": np.zeros((10, 10), dtype=np.float32),
        "template": np.zeros((5, 5), dtype=np.uint8),
    }
    return LogoDetector(profile)


def feed(det, raw_state, n, start=1000.0, dt=0.1):
    """Drive _commit_state n times, returning the final state."""
    now = start
    state = det.state
    for _ in range(n):
        state = det._commit_state(raw_state, now)
        now += dt
    return state


def test_enter_requires_debounce_frames():
    det = make_logo_detector()
    # One short of DEBOUNCE_FRAMES: still WATCHING.
    feed(det, DetectionState.COMMERCIAL, config.DEBOUNCE_FRAMES - 1)
    assert det.state == DetectionState.WATCHING
    # One more crosses the threshold.
    feed(det, DetectionState.COMMERCIAL, 1, start=2000.0)
    assert det.state == DetectionState.COMMERCIAL


def test_exit_is_fast_by_default():
    det = make_logo_detector()
    feed(det, DetectionState.COMMERCIAL, config.DEBOUNCE_FRAMES)
    assert det.state == DetectionState.COMMERCIAL
    # Decay disabled by default -> exit takes only RECOVERY_DEBOUNCE_FRAMES.
    feed(det, DetectionState.WATCHING, config.RECOVERY_DEBOUNCE_FRAMES, start=3000.0)
    assert det.state == DetectionState.WATCHING


def test_required_debounce_matches_today_when_decay_off():
    det = make_logo_detector()
    # WATCHING -> entering: DEBOUNCE_FRAMES (no cooldown active).
    assert det._required_debounce(1000.0) == config.DEBOUNCE_FRAMES


def test_set_break_decay_toggles_flag():
    det = make_logo_detector()
    assert det._break_decay_enabled is False
    det.set_break_decay(True)
    assert det._break_decay_enabled is True
    det.set_break_decay(False)
    assert det._break_decay_enabled is False


def _enter_break(det, enter_t):
    """Put the detector into COMMERCIAL with a known break start time."""
    feed(det, DetectionState.COMMERCIAL, config.DEBOUNCE_FRAMES, start=enter_t, dt=0.0)
    assert det.state == DetectionState.COMMERCIAL
    det._commercial_enter_t = enter_t  # pin it exactly for deterministic t


def test_decay_adds_offgrid_exit_friction():
    det = make_logo_detector()
    det.set_break_decay(True)
    _enter_break(det, 1000.0)
    # now = 1045 -> t = 45s (off grid midpoint) -> full extra frames.
    extra = round(config.EXIT_DECAY_MAX_FRAMES * config.BREAK_DECAY_STRENGTH)
    assert det._required_debounce(1045.0) == config.RECOVERY_DEBOUNCE_FRAMES + extra


def test_decay_disabled_is_baseline_exit():
    det = make_logo_detector()
    det.set_break_decay(False)
    _enter_break(det, 1000.0)
    assert det._required_debounce(1045.0) == config.RECOVERY_DEBOUNCE_FRAMES


def test_offgrid_blip_is_swallowed_but_sustained_return_exits():
    det = make_logo_detector()
    det.set_break_decay(True)
    _enter_break(det, 1000.0)
    extra = round(config.EXIT_DECAY_MAX_FRAMES * config.BREAK_DECAY_STRENGTH)
    required = config.RECOVERY_DEBOUNCE_FRAMES + extra

    # A 2-frame "game's back!" blip at off grid t -> swallowed.
    now = 1045.0
    det._commit_state(DetectionState.WATCHING, now); now += 0.1
    det._commit_state(DetectionState.WATCHING, now); now += 0.1
    det._commit_state(DetectionState.COMMERCIAL, now); now += 0.1
    assert det.state == DetectionState.COMMERCIAL

    # A sustained return clears the (capped) requirement -> exits.
    det._commercial_enter_t = 1000.0  # keep t off grid as frames advance
    for _ in range(required + 1):
        det._commit_state(DetectionState.WATCHING, now); now += 0.1
    assert det.state == DetectionState.WATCHING


def _arm_recovering(det, monkeypatch):
    """COMMERCIAL state, decay on, debounce already satisfied, confidence sitting
    BETWEEN the match and recovery thresholds (fails recovery bar, passes the
    eased match bar)."""
    det.set_break_decay(True)
    det._current_state = DetectionState.COMMERCIAL
    det._candidate_state = DetectionState.WATCHING
    det._consecutive_count = 999          # debounce won't block the transition
    det._commercial_enter_t = 1000.0
    mid = (config.LOGO_MATCH_THRESHOLD + config.LOGO_RECOVERY_THRESHOLD) / 2
    monkeypatch.setattr(det, "_edge_confidence", lambda g: 0.0)
    monkeypatch.setattr(det, "_template_confidence", lambda g: mid)


def test_past_horizon_eases_exit_threshold(monkeypatch):
    det = make_logo_detector()
    _arm_recovering(det, monkeypatch)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    det.check_frame(frame, now=1130.0)    # t=130 -> past 120 -> ease -> exits
    assert det.state == DetectionState.WATCHING


def test_offgrid_does_not_ease_exit_threshold(monkeypatch):
    det = make_logo_detector()
    _arm_recovering(det, monkeypatch)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    det.check_frame(frame, now=1045.0)    # t=45 off grid -> recovery bar -> stays
    assert det.state == DetectionState.COMMERCIAL
