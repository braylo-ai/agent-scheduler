"""Open-loop load test against the SIMULATED MockLLM.

One heavy tenant offers 4x the load of three light tenants. Every config runs the same
pre-drawn arrivals and call trees. Output: results/loadtest.json + results/loadtest.md.

    python -m agent_scheduler.loadtest            # full run (~2.5 min)
    python -m agent_scheduler.loadtest --quick    # smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .admission import AdmissionController
from .core import CallContext
from .metrics import jain, normalized_jain, percentile
from .mock_llm import MockLLM
from .queues import FairQueue, FifoQueue
from .scheduler import DeadlineExceeded, JobCancelled, Rejected, Scheduler
from .workload import make_tree, run_agent


@dataclass
class Config:
    name: str
    fair: bool
    deadline_propagation: bool
    cancel_tree: bool
    admission: bool


CONFIGS = [
    Config(
        "fifo_naive",
        fair=False,
        deadline_propagation=False,
        cancel_tree=False,
        admission=False,
    ),
    Config(
        "fair_naive",
        fair=True,
        deadline_propagation=False,
        cancel_tree=False,
        admission=False,
    ),
    Config(
        "fair_deadline",
        fair=True,
        deadline_propagation=True,
        cancel_tree=True,
        admission=False,
    ),
    Config(
        "fifo_full",
        fair=False,
        deadline_propagation=True,
        cancel_tree=True,
        admission=True,
    ),
    Config(
        "full", fair=True, deadline_propagation=True, cancel_tree=True, admission=True
    ),
]


def make_arrivals(seed: int, duration_s: float, rates: dict[str, float]):
    rng = random.Random(seed)
    arr = []
    for tenant, rate in rates.items():
        t = 0.0
        while True:
            t += rng.expovariate(rate)
            if t >= duration_s:
                break
            arr.append((t, tenant))
    arr.sort()
    tree_rng = random.Random(seed + 1)
    return [(t, ten, make_tree(tree_rng)) for t, ten in arr]


async def run_config(
    cfg: Config, arrivals, slots: int, slo_s: float, duration_s: float, seed: int
) -> dict:
    work_calls = sum(tr.n_calls() for _, _, tr in arrivals) / max(len(arrivals), 1)
    llm = MockLLM(seed=seed)
    queue = FairQueue() if cfg.fair else FifoQueue()
    adm = (
        AdmissionController(
            slots, max_queue=8 * slots, max_inflight_per_tenant=24, work_calls=work_calls, fair=cfg.fair
        )
        if cfg.admission
        else None
    )
    sched = Scheduler(llm, slots=slots, queue=queue, admission=adm)
    records = []
    loop = asyncio.get_running_loop()

    async def root(tenant: str, tree) -> None:
        ctx = CallContext.root(
            tenant, slo_s, propagate_deadline=cfg.deadline_propagation
        )
        ok, reason = sched.try_admit(ctx)
        if not ok:
            records.append(
                {
                    "tenant": tenant,
                    "status": "rejected",
                    "reason": reason,
                    "latency": None,
                    "calls": 0,
                }
            )
            return
        timer = (
            loop.call_later(slo_s, ctx.scope.cancel, "root_deadline")
            if cfg.cancel_tree
            else None
        )
        t0 = time.monotonic()
        status = "ok"
        try:
            await run_agent(sched, ctx, tree, cancel_siblings=cfg.cancel_tree)
            if time.monotonic() - t0 > slo_s:
                status = "late"
        except (DeadlineExceeded, JobCancelled):
            status = "deadline_exceeded"
        except Rejected:
            status = "rejected"
        finally:
            if timer:
                timer.cancel()
            sched.release(ctx)
        records.append(
            {
                "tenant": tenant,
                "status": status,
                "latency": time.monotonic() - t0,
                "calls": ctx.meter["calls"],
            }
        )

    async with sched:
        start = time.monotonic()
        tasks = []
        for t, tenant, tree in arrivals:
            delay = start + t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(root(tenant, tree)))
        await asyncio.gather(*tasks)
        wall = time.monotonic() - start

    tenants = sorted({ten for _, ten, _ in arrivals})
    offered = Counter(ten for _, ten, _ in arrivals)
    by_status = Counter(r["status"] for r in records)
    ok_lat = [r["latency"] for r in records if r["status"] == "ok"]
    done_lat = [r["latency"] for r in records if r["status"] in ("ok", "late")]
    good = {
        t: sum(1 for r in records if r["tenant"] == t and r["status"] == "ok")
        / duration_s
        for t in tenants
    }
    demand = {t: offered[t] / duration_s for t in tenants}
    n = len(records)
    admitted = n - by_status["rejected"]
    useful_calls = sum(r["calls"] for r in records if r["status"] == "ok")
    total_calls = sum(r["calls"] for r in records)
    per_tenant = {
        t: {
            "offered": offered[t],
            "ok": sum(1 for r in records if r["tenant"] == t and r["status"] == "ok"),
            "ok_rate": round(
                sum(1 for r in records if r["tenant"] == t and r["status"] == "ok")
                / max(offered[t], 1),
                4,
            ),
        }
        for t in tenants
    }
    return {
        "config": asdict(cfg),
        "requests": n,
        "status": dict(by_status),
        "on_time_rate": round(by_status["ok"] / n, 4),
        "deadline_miss_rate_admitted": round(
            (by_status["late"] + by_status["deadline_exceeded"]) / max(admitted, 1), 4
        ),
        "deadline_miss_or_reject_rate": round((n - by_status["ok"]) / n, 4),
        "p50_ok_s": round(percentile(ok_lat, 50), 4),
        "p95_ok_s": round(percentile(ok_lat, 95), 4),
        "p95_all_completed_s": round(percentile(done_lat, 95), 4),
        "goodput_rps": round(by_status["ok"] / duration_s, 3),
        "jain_normalized_goodput": round(normalized_jain(good, demand), 4),
        "jain_raw_ok_rate": round(jain([per_tenant[t]["ok_rate"] for t in tenants]), 4),
        "per_tenant": per_tenant,
        "llm_calls_completed": total_calls,
        "llm_calls_wasted": total_calls - useful_calls,
        "wasted_call_fraction": round(
            (total_calls - useful_calls) / max(total_calls, 1), 4
        ),
        "llm_calls_cancelled_mid_flight": llm.cancelled,
        "scheduler_stats": dict(sched.stats),
        "wall_s": round(wall, 2),
    }


def to_markdown(res: dict) -> str:
    lines = [
        f"Load: {res['load_label']} | SIMULATED MockLLM | seed {res['seed']} | slots {res['slots']} | "
        f"SLO {res['slo_s']}s | {res['duration_s']}s arrivals",
        "",
        "| config | on-time | miss (admitted) | miss+reject | p50 ok (s) | p95 ok (s) | Jain (norm.) | wasted calls |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res["runs"]:
        lines.append(
            f"| {r['config']['name']} | {r['on_time_rate']:.3f} | {r['deadline_miss_rate_admitted']:.3f} | "
            f"{r['deadline_miss_or_reject_rate']:.3f} | {r['p50_ok_s']:.3f} | {r['p95_ok_s']:.3f} | "
            f"{r['jain_normalized_goodput']:.3f} | {r['wasted_call_fraction']:.3f} |"
        )
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    slots, slo = 8, 1.5
    duration = 2.0 if a.quick else 10.0
    loads = {
        "high_21rps": {"heavy": 12.0, "light1": 3.0, "light2": 3.0, "light3": 3.0},
        "low_8.4rps": {"heavy": 4.8, "light1": 1.2, "light2": 1.2, "light3": 1.2},
    }
    out = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "note": "SIMULATED LLM latency (MockLLM); real asyncio scheduling",
        "loads": [],
    }
    md = []
    for label, rates in loads.items():
        arrivals = make_arrivals(a.seed, duration, rates)
        runs = []
        for cfg in CONFIGS:
            r = asyncio.run(run_config(cfg, arrivals, slots, slo, duration, a.seed))
            runs.append(r)
            print(
                f"[{label}] {cfg.name}: on_time={r['on_time_rate']} jain={r['jain_normalized_goodput']} "
                f"p95={r['p95_ok_s']} wasted={r['wasted_call_fraction']}",
                flush=True,
            )
        res = {
            "load_label": label,
            "rates": rates,
            "seed": a.seed,
            "slots": slots,
            "slo_s": slo,
            "duration_s": duration,
            "runs": runs,
        }
        out["loads"].append(res)
        md.append(to_markdown(res))
    if not a.quick:
        p = Path(a.out)
        p.mkdir(exist_ok=True)
        (p / "loadtest.json").write_text(json.dumps(out, indent=2))
        (p / "loadtest.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
