"""Fixture test file (never executed by the eval)."""

from app.core.rate_limiter import RateLimiter


def test_burst():
    limiter = RateLimiter(burst=2)
    assert limiter.allow("k")
    assert limiter.allow("k")
    assert not limiter.allow("k")
