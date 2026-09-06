"""
Pure input validation for study_control_panel.py.

No rclpy, no Tornado — kept separate and tested the same way as every
other bit of logic in this package, even though the panel itself is a
thin orchestration layer around already-validated CLI commands.
"""

VALID_ENV_IDS = ('env_a', 'env_b')


def is_valid_user_id(user_id: str) -> bool:
    """Return True if user_id is a non-empty, digits-only participant ID."""
    return isinstance(user_id, str) and user_id.strip() != '' and user_id.strip().isdigit()


def is_valid_env_id(env_id: str) -> bool:
    return env_id in VALID_ENV_IDS


def normalize_user_id(user_id: str) -> str:
    """Assumes is_valid_user_id(user_id) is already True."""
    return user_id.strip()
