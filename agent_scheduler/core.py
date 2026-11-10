"""Deadlines, cancellation trees and call contexts.

A CallContext carries (tenant, deadline, cancel scope, meter) through every
sub-agent call. Children inherit the parent's deadline minus a reserve, and a
cancel on any scope cascades to every descendant scope.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


class Deadline:
    """Absolute deadline on a monotonic clock. math.inf = unbounded."""

    __slots__ = ("at", "clock")

    def __init__(
        self, at: float = math.inf, clock: Callable[[], float] = time.monotonic
    ):
        self.at = at
        self.clock = clock

    @classmethod
    def after(
        cls, seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> "Deadline":
        return cls(clock() + seconds, clock)

    @classmethod
    def none(cls, clock: Callable[[], float] = time.monotonic) -> "Deadline":
        return cls(math.inf, clock)

    @property
    def bounded(self) -> bool:
        return math.isfinite(self.at)

    def remaining(self) -> float:
        return self.at - self.clock()

    def expired(self) -> bool:
        return self.bounded and self.remaining() <= 0

    def child(self, budget: Optional[float] = None, reserve: float = 0.0) -> "Deadline":
        """Child deadline = min(parent - reserve, now + budget). Never later than parent."""
        at = self.at - reserve
        if budget is not None:
            at = min(at, self.clock() + budget)
        return Deadline(at, self.clock)


class CancelScope:
    """Node in a cancellation tree. cancel() cascades to all descendants."""

    def __init__(self, parent: Optional["CancelScope"] = None, name: str = ""):
        self.parent = parent
        self.name = name
        self.children: list[CancelScope] = []
        self.cancelled = False
        self.reason: Optional[str] = None
        self._callbacks: list[Callable[[str], None]] = []
        if parent is not None:
            parent.children.append(self)
            if parent.cancelled:
                self.cancel(parent.reason or "parent_cancelled")

    def child(self, name: str = "") -> "CancelScope":
        return CancelScope(self, name)

    def cancel(self, reason: str = "cancelled") -> int:
        """Cancel this scope and every descendant. Returns number of scopes newly cancelled."""
        if self.cancelled:
            return 0
        self.cancelled = True
        self.reason = reason
        for cb in list(self._callbacks):
            cb(reason)
        self._callbacks.clear()
        return 1 + sum(c.cancel(reason) for c in list(self.children))

    def on_cancel(self, cb: Callable[[str], None]) -> Callable[[], None]:
        """Register a callback. Returns an unregister function."""
        if self.cancelled:
            cb(self.reason or "cancelled")
            return lambda: None
        self._callbacks.append(cb)

        def unregister() -> None:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

        return unregister

    def close(self) -> None:
        """Detach from parent once finished, so long-lived roots do not leak children."""
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


@dataclass
class CallContext:
    tenant: str
    deadline: Deadline
    scope: CancelScope
    propagate_deadline: bool = True
    meter: dict = field(default_factory=lambda: {"calls": 0, "tokens": 0})

    @classmethod
    def root(
        cls,
        tenant: str,
        slo_s: float,
        propagate_deadline: bool = True,
        clock: Callable[[], float] = time.monotonic,
        name: str = "root",
    ) -> "CallContext":
        return cls(
            tenant,
            Deadline.after(slo_s, clock),
            CancelScope(name=name),
            propagate_deadline,
        )

    def child(
        self, name: str = "", budget: Optional[float] = None, reserve: float = 0.0
    ) -> "CallContext":
        if self.propagate_deadline:
            dl = self.deadline.child(budget, reserve)
        else:
            dl = Deadline.none(self.deadline.clock)
        return CallContext(
            self.tenant, dl, self.scope.child(name), self.propagate_deadline, self.meter
        )
