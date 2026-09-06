from kinova_gen3_lite_rl_bridge.gripper_toggle import toggle_button_rising_edge


def test_rising_edge_on_first_press():
    edge, pressed = toggle_button_rising_edge([1, 0, 0, 0, 0, 0, 0, 0], was_pressed=False)

    assert edge is True
    assert pressed is True


def test_no_edge_while_held():
    edge, pressed = toggle_button_rising_edge([1, 0, 0, 0, 0, 0, 0, 0], was_pressed=True)

    assert edge is False
    assert pressed is True


def test_no_edge_when_not_pressed():
    edge, pressed = toggle_button_rising_edge([0, 0, 0, 0, 0, 0, 0, 0], was_pressed=False)

    assert edge is False
    assert pressed is False


def test_no_edge_on_release():
    edge, pressed = toggle_button_rising_edge([0, 0, 0, 0, 0, 0, 0, 0], was_pressed=True)

    assert edge is False
    assert pressed is False


def test_short_buttons_array_treated_as_not_pressed():
    edge, pressed = toggle_button_rising_edge([], was_pressed=False)

    assert edge is False
    assert pressed is False
