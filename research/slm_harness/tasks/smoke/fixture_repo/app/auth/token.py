"""Token issuing and validation."""

import hashlib
import time

TOKEN_TTL_SECONDS = 900


class TokenValidator:
    """Validates opaque API tokens against the signing secret."""

    def __init__(self, secret: str) -> None:
        self._secret = secret

    def _sign(self, payload: str) -> str:
        return hashlib.sha256((payload + self._secret).encode()).hexdigest()


def validate_token(token: str, secret: str) -> bool:
    """Return True when the token is well-formed, signed, and unexpired."""
    try:
        payload, issued_at, signature = token.rsplit(":", 2)
    except ValueError:
        return False
    validator = TokenValidator(secret)
    if validator._sign(f"{payload}:{issued_at}") != signature:
        return False
    return (time.time() - float(issued_at)) < TOKEN_TTL_SECONDS
