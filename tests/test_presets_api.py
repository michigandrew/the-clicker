import pytest

import config
from fastapi.testclient import TestClient
import server

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def restore_config():
    """Presets mutate the shared config module, snapshot and restore so these
    tests don't bleed into other test files."""
    saved = {k: getattr(config, k) for k in server.TUNABLE_SETTINGS}
    yield
    for k, v in saved.items():
        setattr(config, k, v)


ALL_PRESET_KEYS = {
    "DEBOUNCE_FRAMES", "LOGO_MATCH_THRESHOLD", "LOGO_RECOVERY_THRESHOLD",
    "RECOVERY_DEBOUNCE_FRAMES", "COLOR_MATCH_THRESHOLD", "COLOR_RECOVERY_THRESHOLD",
    "COLOR_DEBOUNCE_FRAMES", "REENTRY_COOLDOWN_SECONDS", "BREAK_DECAY_STRENGTH",
}


def test_aggressive_applies_bundle():
    r = client.post("/api/preset", json={"name": "aggressive"})
    assert r.status_code == 200
    assert config.DEBOUNCE_FRAMES == 2
    assert config.BREAK_DECAY_STRENGTH == 1.0


def test_conservative_disables_decay():
    r = client.post("/api/preset", json={"name": "conservative"})
    assert r.status_code == 200
    assert config.BREAK_DECAY_STRENGTH == 0.0
    assert config.DEBOUNCE_FRAMES == 5


def test_every_preset_sets_every_key():
    for name in ("conservative", "balanced", "aggressive"):
        assert ALL_PRESET_KEYS.issubset(server.PRESETS[name].keys())
        # every key is a known tunable setting
        assert ALL_PRESET_KEYS.issubset(set(server.TUNABLE_SETTINGS.keys()))


def test_unknown_preset_is_400():
    r = client.post("/api/preset", json={"name": "nope"})
    assert r.status_code == 400


def test_match_thresholds_increase_conservative_to_aggressive():
    """A HIGHER match threshold makes the detector enter COMMERCIAL more eagerly
    (raw_state=COMMERCIAL when confidence < threshold). So the eager to mute
    Aggressive preset must have the HIGHEST match thresholds and Conservative the
    lowest. Guards against the inverted direction this bug originally shipped."""
    c, b, a = server.PRESETS["conservative"], server.PRESETS["balanced"], server.PRESETS["aggressive"]
    for key in ("LOGO_MATCH_THRESHOLD", "COLOR_MATCH_THRESHOLD"):
        assert c[key] < b[key] < a[key], f"{key} not monotonic conservative<balanced<aggressive"


def test_every_preset_preserves_hysteresis_gap():
    """Recovery threshold must exceed match threshold in every preset, or the
    detector oscillates at the boundary."""
    for name in ("conservative", "balanced", "aggressive"):
        p = server.PRESETS[name]
        assert p["LOGO_RECOVERY_THRESHOLD"] > p["LOGO_MATCH_THRESHOLD"], name
        assert p["COLOR_RECOVERY_THRESHOLD"] > p["COLOR_MATCH_THRESHOLD"], name
