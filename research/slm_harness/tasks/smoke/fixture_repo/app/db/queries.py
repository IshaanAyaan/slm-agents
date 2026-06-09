"""Query helpers over the in-memory store."""

from app.db.models import OrderRecord

_ORDERS: list[OrderRecord] = []


def insert_order(order: OrderRecord) -> None:
    """Add an order to the store."""
    _ORDERS.append(order)


def fetch_orders_by_user(user_id: int) -> list[OrderRecord]:
    """Return all orders belonging to one user, newest first."""
    matching = [o for o in _ORDERS if o.user_id == user_id]
    return sorted(matching, key=lambda o: o.order_id, reverse=True)
