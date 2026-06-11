"""Server-side session bookkeeping."""

import time

SESSION_TIMEOUT_SECONDS = 3600

_sessions: dict[str, float] = {}


def open_session(session_id: str) -> None:
    """Register a new session."""
    _sessions[session_id] = time.time()


def refresh_session(session_id: str) -> bool:
    """Extend a live session; False if it already timed out."""
    started = _sessions.get(session_id)
    if started is None or time.time() - started > SESSION_TIMEOUT_SECONDS:
        _sessions.pop(session_id, None)
        return False
    _sessions[session_id] = time.time()
    return True
