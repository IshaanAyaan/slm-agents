"""Persistence models."""

from dataclasses import dataclass, field


@dataclass
class UserRecord:
    """A registered user."""

    user_id: int
    email: str
    is_active: bool = True


@dataclass
class OrderRecord:
    """A single order placed by a user."""

    order_id: int
    user_id: int
    total_cents: int
    items: list[str] = field(default_factory=list)
