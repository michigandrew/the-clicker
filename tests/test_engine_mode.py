from engine import Engine
from intervention import PlaybackMode


class StubDetector:
    def __init__(self):
        self.decay = None

    def set_break_decay(self, enabled):
        self.decay = enabled


def test_live_mode_enables_decay():
    e = Engine()
    e._detector = StubDetector()
    e._set_intervention_mode(PlaybackMode.LIVE)
    assert e._mode == PlaybackMode.LIVE
    assert e._detector.decay is True


def test_delayed_mode_disables_decay():
    e = Engine()
    e._detector = StubDetector()
    e._set_intervention_mode(PlaybackMode.DELAYED)
    assert e._detector.decay is False


def test_none_mode_disables_decay():
    e = Engine()
    e._detector = StubDetector()
    e._set_intervention_mode(None)
    assert e._mode is None
    assert e._detector.decay is False
