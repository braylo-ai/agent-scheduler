"""Agent workload: planner call -> fan-out of sub-agents (each a short tool loop,
sometimes with nested sub-agents) -> synthesis call.

The call tree (token counts + latency jitter) is drawn up front from a seeded RNG, so
every scheduler config executes exactly the same work.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

from .core import CallContext
from .scheduler import Scheduler


@dataclass
class Call:
    prompt: int
    out: int
    jitter: float


@dataclass
class SubAgent:
    calls: list[Call]
    subs: list["SubAgent"] = field(default_factory=list)


@dataclass
class AgentTree:
    plan: Call
    subs: list[SubAgent]
    synth: Call

    def n_calls(self) -> int:
        def count(s: SubAgent) -> int:
            return len(s.calls) + sum(count(c) for c in s.subs)

        return 2 + sum(count(s) for s in self.subs)


def _call(rng: random.Random, prompt=(300, 1200), out=(40, 160), sigma=0.35) -> Call:
    return Call(rng.randint(*prompt), rng.randint(*out), rng.lognormvariate(0.0, sigma))


def make_tree(
    rng: random.Random, fanout=(2, 4), sub_calls=(1, 3), nest_prob=0.2
) -> AgentTree:
    def sub(depth: int) -> SubAgent:
        s = SubAgent([_call(rng) for _ in range(rng.randint(*sub_calls))])
        if depth < 2 and rng.random() < nest_prob:
            s.subs = [sub(depth + 1) for _ in range(2)]
        return s

    return AgentTree(
        _call(rng, out=(60, 200)),
        [sub(1) for _ in range(rng.randint(*fanout))],
        _call(rng, prompt=(800, 2000), out=(100, 300)),
    )


async def _fan_out(coros_ctx, cancel_siblings: bool) -> None:
    """Run sub-agents concurrently. On first failure, cancel sibling scopes (fail fast)."""
    tasks = [asyncio.create_task(c) for c, _ in coros_ctx]
    ctxs = [x for _, x in coros_ctx]
    if not tasks:
        return
    if cancel_siblings:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        excs = [t.exception() for t in done if not t.cancelled()]
        first = next((e for e in excs if e is not None), None)
        if first is not None:
            for c in ctxs:
                c.scope.cancel("sibling_failed")
            await asyncio.gather(*pending, return_exceptions=True)
            raise first
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r


async def run_sub(
    sched: Scheduler,
    ctx: CallContext,
    spec: SubAgent,
    reserve_s: float,
    cancel_siblings: bool,
) -> None:
    try:
        half = max(1, len(spec.calls) // 2)
        for c in spec.calls[:half]:
            await sched.submit(ctx.child("tool"), c.prompt, c.out, c.jitter)
        if spec.subs:
            kids = [
                ctx.child(f"nested{i}", reserve=reserve_s)
                for i in range(len(spec.subs))
            ]
            await _fan_out(
                [
                    (run_sub(sched, k, s, reserve_s, cancel_siblings), k)
                    for k, s in zip(kids, spec.subs)
                ],
                cancel_siblings,
            )
        for c in spec.calls[half:]:
            await sched.submit(ctx.child("tool"), c.prompt, c.out, c.jitter)
    finally:
        ctx.scope.close()


async def run_agent(
    sched: Scheduler,
    ctx: CallContext,
    tree: AgentTree,
    reserve_s: float = 0.15,
    cancel_siblings: bool = True,
) -> None:
    await sched.submit(
        ctx.child("plan"), tree.plan.prompt, tree.plan.out, tree.plan.jitter
    )
    kids = [ctx.child(f"sub{i}", reserve=reserve_s) for i in range(len(tree.subs))]
    await _fan_out(
        [
            (run_sub(sched, k, s, reserve_s, cancel_siblings), k)
            for k, s in zip(kids, tree.subs)
        ],
        cancel_siblings,
    )
    await sched.submit(
        ctx.child("synth"), tree.synth.prompt, tree.synth.out, tree.synth.jitter
    )
