from detector import exit_friction

B = [30, 60, 90, 120]
W = 10
MAXF = 20  # explicit so the test is independent of config


def f(t, strength=1.0):
    return exit_friction(t, strength, B, W, MAXF)


def test_near_boundary_is_baseline():
    assert f(30) == (0, False)
    assert f(60) == (0, False)
    assert f(25) == (0, False)   # within window of 30
    assert f(68) == (0, False)   # within window of 60


def test_offgrid_midpoint_is_max():
    # midpoint of a 30s gap is 15s from either boundary -> full friction
    assert f(45) == (MAXF, False)
    assert f(75) == (MAXF, False)


def test_early_break_is_max_friction():
    # 2s into a break: a "break ending" now is absurd -> max friction
    assert f(2) == (MAXF, False)


def test_past_longest_horizon_eases_with_no_friction():
    assert f(120) == (0, True)
    assert f(150) == (0, True)


def test_strength_scales_and_zero_disables():
    assert f(45, strength=0.5) == (round(MAXF * 0.5), False)
    assert f(45, strength=0.0) == (0, False)
    assert f(2, strength=0.0) == (0, False)


def test_never_exceeds_cap():
    for t in range(0, 120):
        extra, _ = f(t, strength=1.0)
        assert 0 <= extra <= MAXF


def test_empty_boundaries_is_noop():
    assert exit_friction(45, 1.0, [], W, MAXF) == (0, False)
