"""The capture card is shared with the overnight capture rig; a hold keeps
The Clicker off it for a while, and expires on its own."""
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from engine import Engine, EngineState


def test_no_hold_by_default():
    assert Engine().capture_held() is False
    assert Engine().hold_expiry() is None


def test_hold_refuses_to_start_the_engine():
    e = Engine()
    e.hold_capture(3600, "gambling-meter rig")
    assert e.capture_held()
    with pytest.raises(RuntimeError, match="gambling-meter rig"):
        asyncio.run(e.start(channel="espn", shadow=True))
    assert e.state.value == "stopped"


def test_release_clears_the_hold():
    e = Engine()
    e.hold_capture(3600)
    e.release_capture()
    assert e.capture_held() is False


def test_hold_expires_on_its_own(monkeypatch):
    e = Engine()
    e.hold_capture(10)
    monkeypatch.setattr(time, "time", lambda: e._hold_until + 1)
    assert e.capture_held() is False


def test_status_reports_the_expiry():
    e = Engine()
    until = e.hold_capture(60)
    assert e.get_status().capture_held_until == until


def test_running_loop_releases_the_device_on_hold():
    """A hold placed while the engine is already running must actually close
    the device, not just refuse future start() calls (regression: the loop
    used to keep /dev/video0 open the whole time a hold was in force, so the
    borrowing process still got "device busy")."""
    e = Engine()
    e._state = EngineState.RUNNING
    e._shadow = True
    e._detector = None
    mock_cap = MagicMock()
    e._cap = mock_cap
    e.hold_capture(3600, "gambling-meter rig")

    async def run_briefly():
        task = asyncio.create_task(e._loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_briefly())

    mock_cap.close.assert_called_once()
    assert e._cap is None


def test_running_loop_reopens_the_device_once_hold_clears():
    e = Engine()
    e._state = EngineState.RUNNING
    e._shadow = True
    e._detector = None
    e._cap = None
    e.hold_capture(3600)
    e.release_capture()  # simulate the hold having already cleared

    with patch("engine.FrameCapture") as MockFrameCapture:
        mock_cap = MagicMock()
        mock_cap.grab_frame.return_value = None
        MockFrameCapture.return_value = mock_cap

        async def run_briefly():
            task = asyncio.create_task(e._loop())
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_briefly())

        mock_cap.open.assert_called_once()
        assert e._cap is mock_cap
