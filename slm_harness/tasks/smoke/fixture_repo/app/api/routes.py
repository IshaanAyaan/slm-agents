"""HTTP route registration."""

from app.core.rate_limiter import RateLimiter
from app.db.queries import fetch_orders_by_user

_limiter = RateLimiter()


def register_routes(router) -> None:
    """Attach all HTTP handlers to the router."""
    router.get("/orders/{user_id}", _list_orders)


def _list_orders(request):
    if not _limiter.allow(request.client_ip):
        return {"status": 429}
    return {"status": 200, "orders": fetch_orders_by_user(request.path_params["user_id"])}
