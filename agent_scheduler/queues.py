"""Queues: FIFO baseline and deficit-round-robin fair queue (EDF inside each tenant)."""

from __future__ import annotations

import heapq
from collections import Counter, deque
from typing import Optional


class FifoQueue:
    """Global first-in-first-out. Baseline: one noisy tenant starves the rest."""

    fair = False

    def __init__(self) -> None:
        self._q: deque = deque()
        self._tlen: Counter = Counter()
        self._n = 0

    def push(self, job) -> None:
        self._q.append(job)
        self._tlen[job.tenant] += 1
        self._n += 1

    def discard(self, job) -> None:
        if not getattr(job, "removed", False) and getattr(job, "queued", True):
            job.removed = True
            self._tlen[job.tenant] -= 1
            self._n -= 1

    def pop(self):
        while self._q:
            job = self._q.popleft()
            if getattr(job, "removed", False):
                continue
            self._tlen[job.tenant] -= 1
            self._n -= 1
            job.queued = False
            return job
        return None

    def __len__(self) -> int:
        return self._n

    def tenant_len(self, tenant: str) -> int:
        return self._tlen[tenant]

    def active_tenants(self) -> int:
        return sum(1 for v in self._tlen.values() if v > 0)


class FairQueue:
    """Deficit round robin across tenants, weighted; earliest-deadline-first within a tenant.

    Each job has a `cost` (tokens). A tenant at the head of the round may dequeue while its
    deficit covers the head job's cost; otherwise it gets `quantum * weight` more credit and
    moves to the back. Long-run share of cost served is proportional to weight.
    """

    fair = True

    def __init__(self, quantum: float = 256, weights: Optional[dict] = None) -> None:
        self.quantum = quantum
        self.weights = weights or {}
        self._heaps: dict[str, list] = {}
        self._deficit: Counter = Counter()
        self._active: deque = deque()
        self._in_active: set = set()
        self._tlen: Counter = Counter()
        self._n = 0
        self._seq = 0

    def push(self, job) -> None:
        h = self._heaps.setdefault(job.tenant, [])
        self._seq += 1
        heapq.heappush(h, (job.deadline.at, self._seq, job))
        if job.tenant not in self._in_active:
            self._active.append(job.tenant)
            self._in_active.add(job.tenant)
        self._tlen[job.tenant] += 1
        self._n += 1

    def discard(self, job) -> None:
        if not getattr(job, "removed", False) and getattr(job, "queued", True):
            job.removed = True
            self._tlen[job.tenant] -= 1
            self._n -= 1

    def _clean_head(self, h: list) -> None:
        while h and getattr(h[0][2], "removed", False):
            heapq.heappop(h)

    def _retire(self, tenant: str) -> None:
        self._active.popleft()
        self._in_active.discard(tenant)
        self._deficit[tenant] = 0

    def pop(self):
        while self._active:
            t = self._active[0]
            h = self._heaps.get(t, [])
            self._clean_head(h)
            if not h:
                self._retire(t)
                continue
            job = h[0][2]
            if self._deficit[t] >= job.cost:
                self._deficit[t] -= job.cost
                heapq.heappop(h)
                self._tlen[t] -= 1
                self._n -= 1
                job.queued = False
                self._clean_head(h)
                if not h:
                    self._retire(t)
                return job
            self._deficit[t] += self.quantum * self.weights.get(t, 1.0)
            self._active.rotate(-1)
        return None

    def __len__(self) -> int:
        return self._n

    def tenant_len(self, tenant: str) -> int:
        return self._tlen[tenant]

    def active_tenants(self) -> int:
        return sum(1 for v in self._tlen.values() if v > 0)
