import asyncio
import math
import random

import pytest

from agent_scheduler.admission import AdmissionController
from agent_scheduler.core import CallContext, CancelScope, Deadline
from agent_scheduler.metrics import jain, max_min_shares, normalized_jain, percentile
from agent_scheduler.mock_llm import MockLLM
from agent_scheduler.queues import FairQueue, FifoQueue
from agent_scheduler.scheduler import DeadlineExceeded, JobCancelled, Scheduler
from agent_scheduler.workload import make_tree, run_agent


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class J:
    def __init__(self, tenant, cost=10, at=math.inf):
        self.tenant, self.cost, self.deadline = tenant, cost, Deadline(at)


# ---- deadlines -----------------------------------------------------------
def test_child_deadline_never_exceeds_parent():
    c = FakeClock()
    d = Deadline(c() + 10, c)
    assert d.child(budget=50).at == d.at
    assert d.child(budget=2).at == c() + 2
    assert d.child(reserve=3).at == d.at - 3


def test_deadline_expiry_and_unbounded():
    c = FakeClock()
    d = Deadline(c() + 1, c)
    assert not d.expired()
    c.t += 1.5
    assert d.expired()
    assert not Deadline.none(c).expired() and not Deadline.none(c).bounded


def test_context_propagation_off_gives_unbounded_children():
    ctx = CallContext.root("t", 1.0, propagate_deadline=False)
    assert ctx.deadline.bounded and not ctx.child().deadline.bounded
    ctx2 = CallContext.root("t", 1.0)
    assert ctx2.child(reserve=0.2).deadline.at == pytest.approx(ctx2.deadline.at - 0.2)


# ---- cancellation tree ----------------------------------------------------
def test_cancel_cascades_to_all_descendants():
    root = CancelScope(name="root")
    a, b = root.child("a"), root.child("b")
    a1, a2 = a.child(), a.child()
    assert root.cancel("x") == 5
    assert all(s.cancelled and s.reason == "x" for s in (a, b, a1, a2))


def test_cancel_child_does_not_touch_parent_or_siblings():
    root = CancelScope()
    a, b = root.child(), root.child()
    a.cancel()
    assert a.cancelled and not root.cancelled and not b.cancelled


def test_child_of_cancelled_parent_starts_cancelled_and_callbacks_fire():
    root = CancelScope()
    seen = []
    root.on_cancel(seen.append)
    root.cancel("deadline")
    assert seen == ["deadline"]
    late = root.child()
    assert late.cancelled
    got = []
    late.on_cancel(got.append)  # already cancelled -> immediate
    assert got == ["deadline"]


def test_close_detaches_child():
    root = CancelScope()
    c = root.child()
    c.close()
    assert root.size() == 1


# ---- queues -----------------------------------------------------------------
def test_fifo_preserves_order_and_discard():
    q = FifoQueue()
    jobs = [J("a"), J("b"), J("a")]
    for j in jobs:
        q.push(j)
    q.discard(jobs[1])
    assert len(q) == 2 and q.tenant_len("b") == 0
    assert q.pop() is jobs[0] and q.pop() is jobs[2] and q.pop() is None


def test_fair_queue_interleaves_noisy_and_quiet_tenant():
    q = FairQueue(quantum=10)
    for _ in range(20):
        q.push(J("noisy"))
    q.push(J("quiet"))
    order = [q.pop().tenant for _ in range(4)]
    assert "quiet" in order[:3]  # FIFO would serve quiet 21st


def test_fair_queue_respects_weights():
    q = FairQueue(quantum=10, weights={"gold": 2.0})
    for _ in range(60):
        q.push(J("gold"))
        q.push(J("std"))
    served = [q.pop().tenant for _ in range(60)]
    ratio = served.count("gold") / served.count("std")
    assert 1.7 <= ratio <= 2.3


def test_fair_queue_edf_within_tenant():
    q = FairQueue(quantum=100)
    late, early = J("a", at=50.0), J("a", at=10.0)
    q.push(late)
    q.push(early)
    assert q.pop() is early


# ---- admission -------------------------------------------------------------
def test_admission_queue_full_and_tenant_limit():
    a = AdmissionController(
        slots=2, max_queue=3, max_inflight_per_tenant=1, deadline_aware=False
    )
    dl = Deadline.none()
    assert a.admit("t", dl, 0, queue_len=3) == (False, "queue_full")
    assert a.admit("t", dl, 0, queue_len=0) == (True, None)
    assert a.admit("t", dl, 0, queue_len=0) == (False, "tenant_limit")
    a.release("t")
    assert a.admit("t", dl, 0, queue_len=0)[0]


def test_admission_rejects_infeasible_deadline():
    c = FakeClock()
    a = AdmissionController(
        slots=1, init_service_s=0.1, critical_path_calls=6, work_calls=10
    )
    assert a.admit("t", Deadline(c() + 0.3, c), c(), 0) == (
        False,
        "deadline_infeasible",
    )
    assert a.admit("t", Deadline(c() + 5.0, c), c(), 0)[0]


def test_fair_admission_isolates_quiet_tenant_from_noisy_backlog():
    a = AdmissionController(slots=4, init_service_s=0.1, work_calls=10, fair=True)
    a.inflight["noisy"] = 40
    fifo = AdmissionController(slots=4, init_service_s=0.1, work_calls=10, fair=False)
    fifo.inflight["noisy"] = 40
    assert a.estimate_completion("quiet") < fifo.estimate_completion("quiet")


# ---- scheduler (real asyncio, mock LLM) ----------------------------------------
def test_submit_completes_and_meters():
    async def go():
        async with Scheduler(MockLLM(ttft_ms=1, decode_ms_per_tok=0), slots=2) as s:
            ctx = CallContext.root("t", 5.0)
            r = await s.submit(ctx.child(), 10, 5, jitter=1.0)
            return r, ctx.meter

    r, meter = asyncio.run(go())
    assert r["out_tokens"] == 5 and meter["calls"] == 1 and meter["tokens"] == 15


def test_cancelling_parent_cancels_queued_and_running_children():
    async def go():
        llm = MockLLM(ttft_ms=200, decode_ms_per_tok=0)
        async with Scheduler(llm, slots=1) as s:
            ctx = CallContext.root("t", 10.0)
            tasks = [
                asyncio.create_task(s.submit(ctx.child(), 1, 1, jitter=1.0))
                for _ in range(3)
            ]
            await asyncio.sleep(0.05)  # first one running, two queued
            ctx.scope.cancel("user_abort")
            res = await asyncio.gather(*tasks, return_exceptions=True)
            return res, llm, s

    res, llm, s = asyncio.run(go())
    assert all(isinstance(r, JobCancelled) for r in res)
    assert llm.cancelled == 1 and llm.completed == 0
    assert len(s.queue) == 0


def test_expired_job_is_shed_in_queue_without_using_a_slot():
    async def go():
        llm = MockLLM(ttft_ms=150, decode_ms_per_tok=0)
        async with Scheduler(llm, slots=1) as s:
            blocker = asyncio.create_task(
                s.submit(CallContext.root("a", 5).child(), 1, 1, jitter=1.0)
            )
            await asyncio.sleep(0.01)
            short = CallContext.root("b", 0.05)
            with pytest.raises(DeadlineExceeded):
                await s.submit(short.child(), 1, 1, jitter=1.0)
            await blocker
            return llm, s

    llm, s = asyncio.run(go())
    assert s.stats["shed_expired_in_queue"] == 1 and llm.calls == 1


def test_running_job_killed_at_deadline():
    async def go():
        llm = MockLLM(ttft_ms=500, decode_ms_per_tok=0)
        async with Scheduler(llm, slots=1) as s:
            with pytest.raises(DeadlineExceeded):
                await s.submit(CallContext.root("a", 0.05).child(), 1, 1, jitter=1.0)
            return llm

    assert asyncio.run(go()).cancelled == 1


def test_agent_sibling_failure_cancels_siblings():
    async def go():
        llm = MockLLM(ttft_ms=20, decode_ms_per_tok=0)
        async with Scheduler(llm, slots=4) as s:
            tree = make_tree(
                random.Random(3), fanout=(4, 4), sub_calls=(3, 3), nest_prob=0
            )
            ctx = CallContext.root("t", 0.12)  # too short: some sub fails
            with pytest.raises((DeadlineExceeded, JobCancelled)):
                await run_agent(s, ctx, tree, reserve_s=0.0)
            return llm, tree

    llm, tree = asyncio.run(go())
    assert llm.completed < tree.n_calls()


def test_agent_completes_with_ample_deadline():
    async def go():
        async with Scheduler(MockLLM(ttft_ms=2, decode_ms_per_tok=0.001), slots=4) as s:
            tree = make_tree(random.Random(1))
            ctx = CallContext.root("t", 30.0)
            await run_agent(s, ctx, tree)
            return ctx.meter["calls"], tree.n_calls()

    done, total = asyncio.run(go())
    assert done == total


# ---- metrics -------------------------------------------------------------------
def test_percentile_and_jain_bounds():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isnan(percentile([], 95))
    assert jain([1, 1, 1, 1]) == 1.0
    assert jain([1, 0, 0, 0]) == pytest.approx(0.25)


def test_max_min_water_filling():
    s = max_min_shares({"a": 1, "b": 10, "c": 10}, 9)
    assert s == pytest.approx({"a": 1, "b": 4, "c": 4})
    assert normalized_jain(
        {"a": 1, "b": 4, "c": 4}, {"a": 1, "b": 10, "c": 10}
    ) == pytest.approx(1.0)
