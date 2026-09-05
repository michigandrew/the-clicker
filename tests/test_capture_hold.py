"""The capture card is shared with the overnight capture rig; a hold keeps
The Clicker off it for a while, and expires on its own."""
import asyncio
import time

import pytest

from engine import Engine


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
