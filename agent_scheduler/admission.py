"""Admission control at the root (agent request) level.

Rejecting at entry is cheap; rejecting a sub-agent call mid-flight wastes the work already
done. So the gate sits in front of the whole agent request and uses three checks:
  1. global queue depth cap,
  2. per-tenant in-flight root cap,
  3. deadline feasibility: estimated completion (critical path, or share of in-flight
     work under processor sharing) > remaining budget -> reject.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional


class AdmissionController:
    def __init__(
        self,
        slots: int,
        max_queue: int = 64,
        max_inflight_per_tenant: Optional[int] = None,
        deadline_aware: bool = True,
        init_service_s: float = 0.08,
        alpha: float = 0.1,
        critical_path_calls: float = 6.0,
        work_calls: float = 10.0,
        fair: bool = True,
    ) -> None:
        self.slots = slots
        self.max_queue = max_queue
        self.max_inflight = max_inflight_per_tenant
        self.deadline_aware = deadline_aware
        self.service_s = init_service_s
        self.alpha = alpha
        self.critical_path_calls = critical_path_calls
        self.work_calls = work_calls
        self.fair = fair
        self.inflight: Counter = Counter()
        self.decisions: Counter = Counter()

    def observe_service(self, seconds: float) -> None:
        self.service_s = (1 - self.alpha) * self.service_s + self.alpha * seconds

    def estimate_completion(self, tenant: str) -> float:
        """Processor-sharing estimate for a new root: its own critical path, or its share of the
        in-flight work, whichever is longer. With fair queueing only roots of tenants that are
        competing for the same share count ahead of it."""
        total = sum(self.inflight.values())
        if self.fair:
            active = sum(1 for v in self.inflight.values() if v > 0) or 1
            if self.inflight[tenant] == 0:
                active += 1
            ahead = min(total, (self.inflight[tenant] + 1) * active - 1)
        else:
            ahead = total
        backlog = (ahead + 1) * self.work_calls * self.service_s / self.slots
        return max(self.critical_path_calls * self.service_s, backlog)

    def admit(self, tenant: str, deadline, now: float, queue_len: int) -> tuple[bool, Optional[str]]:
        reason = None
        if queue_len >= self.max_queue:
            reason = "queue_full"
        elif self.max_inflight is not None and self.inflight[tenant] >= self.max_inflight:
            reason = "tenant_limit"
        elif self.deadline_aware and deadline.bounded:
            if now + self.estimate_completion(tenant) > deadline.at:
                reason = "deadline_infeasible"
        if reason:
            self.decisions[reason] += 1
            return False, reason
        self.inflight[tenant] += 1
        self.decisions["admitted"] += 1
        return True, None

    def release(self, tenant: str) -> None:
        self.inflight[tenant] = max(0, self.inflight[tenant] - 1)
