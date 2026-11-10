"""Async scheduler: N worker slots (provider concurrency) pulling from a queue.

- Jobs whose cancel scope fires are removed from the queue or have their running
  LLM task cancelled.
- Jobs whose deadline passes while queued are shed before they use a slot.
- Jobs whose deadline passes while running are cancelled (no wasted tail work).
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import Counter
from typing import Optional

from .core import CallContext, Deadline, CancelScope
from .queues import FairQueue


class Rejected(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


class JobCancelled(Exception):
    pass


_seq = itertools.count()


class Job:
    __slots__ = (
        "tenant",
        "cost",
        "out_tokens",
        "jitter",
        "deadline",
        "scope",
        "future",
        "enqueued_at",
        "seq",
        "state",
        "task",
        "removed",
        "queued",
        "meter",
    )

    def __init__(
        self,
        tenant: str,
        cost: int,
        out_tokens: int,
        jitter: Optional[float],
        deadline: Deadline,
        scope: CancelScope,
        future,
        now: float,
        meter: dict,
    ):
        self.tenant = tenant
        self.cost = cost
        self.out_tokens = out_tokens
        self.jitter = jitter
        self.deadline = deadline
        self.scope = scope
        self.future = future
        self.enqueued_at = now
        self.seq = next(_seq)
        self.state = "queued"
        self.task: Optional[asyncio.Task] = None
        self.removed = False
        self.queued = True
        self.meter = meter


class Scheduler:
    def __init__(
        self, llm, slots: int = 4, queue=None, admission=None, clock=time.monotonic
    ) -> None:
        self.llm = llm
        self.slots = slots
        self.queue = queue if queue is not None else FairQueue()
        self.admission = admission
        self.clock = clock
        self.stats: Counter = Counter()
        self._cond: Optional[asyncio.Condition] = None
        self._workers: list[asyncio.Task] = []

    async def __aenter__(self) -> "Scheduler":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def start(self) -> None:
        self._cond = asyncio.Condition()
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.slots)]

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    # ---- root-level admission ------------------------------------------------
    def try_admit(self, ctx: CallContext) -> tuple[bool, Optional[str]]:
        if self.admission is None:
            return True, None
        return self.admission.admit(
            ctx.tenant,
            ctx.deadline,
            self.clock(),
            len(self.queue),
        )

    def release(self, ctx: CallContext) -> None:
        if self.admission is not None:
            self.admission.release(ctx.tenant)

    # ---- per-call submit ------------------------------------------------------
    async def submit(
        self,
        ctx: CallContext,
        cost: int,
        out_tokens: int = 64,
        jitter: Optional[float] = None,
    ) -> dict:
        if ctx.scope.cancelled:
            self.stats["cancelled_before_submit"] += 1
            raise JobCancelled(ctx.scope.reason)
        if ctx.deadline.expired():
            self.stats["expired_at_submit"] += 1
            raise DeadlineExceeded("expired_at_submit")
        fut = asyncio.get_running_loop().create_future()
        job = Job(
            ctx.tenant,
            cost,
            out_tokens,
            jitter,
            ctx.deadline,
            ctx.scope,
            fut,
            self.clock(),
            ctx.meter,
        )
        unregister = ctx.scope.on_cancel(lambda reason: self._cancel_job(job, reason))
        if fut.done():  # scope was already cancelled during registration
            unregister()
            return await fut
        async with self._cond:
            self.queue.push(job)
            self.stats["submitted"] += 1
            self._cond.notify()
        try:
            return await fut
        finally:
            unregister()
            if job.state == "queued":
                self.queue.discard(job)
            if job.task is not None and not job.task.done():
                job.task.cancel()

    def _cancel_job(self, job: Job, reason: str) -> None:
        if job.state in ("done", "cancelled"):
            return
        was = job.state
        job.state = "cancelled"
        self.stats[f"cancelled_{was}"] += 1
        if was == "queued":
            self.queue.discard(job)
        if job.task is not None and not job.task.done():
            job.task.cancel()
        if not job.future.done():
            job.future.set_exception(JobCancelled(reason))

    def _pop_runnable(self) -> Optional[Job]:
        while True:
            job = self.queue.pop()
            if job is None:
                return None
            if job.future.done() or job.state != "queued":
                continue
            if job.deadline.expired():
                job.state = "done"
                self.stats["shed_expired_in_queue"] += 1
                job.future.set_exception(DeadlineExceeded("expired_in_queue"))
                continue
            return job

    async def _worker(self) -> None:
        while True:
            async with self._cond:
                job = self._pop_runnable()
                while job is None:
                    await self._cond.wait()
                    job = self._pop_runnable()
            await self._run(job)

    async def _run(self, job: Job) -> None:
        job.state = "running"
        t0 = self.clock()
        job.task = asyncio.create_task(
            self.llm.complete(job.cost, job.out_tokens, job.jitter)
        )
        timeout = job.deadline.remaining() if job.deadline.bounded else None
        done, _ = await asyncio.wait({job.task}, timeout=timeout)
        if not done:
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
            if job.state == "running":
                job.state = "done"
                self.stats["killed_expired_running"] += 1
                if not job.future.done():
                    job.future.set_exception(DeadlineExceeded("expired_while_running"))
            return
        if job.task.cancelled():
            return  # cancelled via scope; future already resolved
        if job.state != "running":
            return
        job.state = "done"
        exc = job.task.exception()
        if exc is not None:
            self.stats["llm_errors"] += 1
            if not job.future.done():
                job.future.set_exception(exc)
            return
        res = job.task.result()
        self.stats["completed"] += 1
        job.meter["calls"] += 1
        job.meter["tokens"] += job.cost + job.out_tokens
        if self.admission is not None:
            self.admission.observe_service(self.clock() - t0)
        res["queue_wait_s"] = t0 - job.enqueued_at
        if not job.future.done():
            job.future.set_result(res)
