from kinova_gen3_lite_rl_bridge.study_session_validation import (
    is_valid_env_id,
    is_valid_user_id,
    normalize_user_id,
)


def test_numeric_user_id_is_valid():
    assert is_valid_user_id('042') is True


def test_empty_user_id_is_invalid():
    assert is_valid_user_id('') is False


def test_whitespace_only_user_id_is_invalid():
    assert is_valid_user_id('   ') is False


def test_non_numeric_user_id_is_invalid():
    assert is_valid_user_id('p42') is False


def test_non_string_user_id_is_invalid():
    assert is_valid_user_id(None) is False
    assert is_valid_user_id(42) is False


def test_user_id_with_surrounding_whitespace_is_valid():
    assert is_valid_user_id('  42  ') is True


def test_normalize_user_id_strips_whitespace():
    assert normalize_user_id('  42  ') == '42'


def test_valid_env_ids():
    assert is_valid_env_id('env_a') is True
    assert is_valid_env_id('env_b') is True


def test_invalid_env_id():
    assert is_valid_env_id('env_c') is False
    assert is_valid_env_id('') is False
