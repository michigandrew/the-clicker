import config


def test_break_decay_constants_exist_with_expected_types():
    assert config.BREAK_BOUNDARIES == [30, 60, 90, 120]
    assert isinstance(config.BREAK_BOUNDARY_WINDOW, (int, float))
    assert config.BREAK_BOUNDARY_WINDOW == 10
    assert config.EXIT_DECAY_MAX_FRAMES == 25
    assert config.BREAK_DECAY_STRENGTH == 0.5
