"""SIMULATED LLM. No model runs: each call sleeps for a latency drawn from
ttft + prefill + decode with lognormal jitter. Numbers produced with it are
scheduler behaviour under a synthetic latency model, not provider measurements."""

from __future__ import annotations

import asyncio
import random
from typing import Optional


class MockLLM:
    def __init__(
        self,
        seed: int = 0,
        ttft_ms: float = 25.0,
        prefill_ms_per_tok: float = 0.02,
        decode_ms_per_tok: float = 0.35,
        sigma: float = 0.35,
    ) -> None:
        self.rng = random.Random(seed)
        self.ttft_ms = ttft_ms
        self.prefill = prefill_ms_per_tok
        self.decode = decode_ms_per_tok
        self.sigma = sigma
        self.calls = 0
        self.completed = 0
        self.cancelled = 0
        self.tokens = 0

    def latency_s(
        self, prompt_tokens: int, out_tokens: int, jitter: Optional[float] = None
    ) -> float:
        if jitter is None:
            jitter = self.rng.lognormvariate(0.0, self.sigma)
        base = self.ttft_ms + prompt_tokens * self.prefill + out_tokens * self.decode
        return base * jitter / 1000.0

    async def complete(
        self, prompt_tokens: int, out_tokens: int, jitter: Optional[float] = None
    ) -> dict:
        lat = self.latency_s(prompt_tokens, out_tokens, jitter)
        self.calls += 1
        try:
            await asyncio.sleep(lat)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.completed += 1
        self.tokens += prompt_tokens + out_tokens
        return {"out_tokens": out_tokens, "latency_s": lat}
