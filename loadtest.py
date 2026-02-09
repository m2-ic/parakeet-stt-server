"""
Load test for Parakeet STT server.

1) 50 sequential requests → average latency
2) Increasing concurrency (1, 2, 3, 5, 8, 10, 15, 20, 30 req/s) for 10 seconds each
   → find where average latency exceeds 1 second

Uses a single ~10s audio file for consistency.
"""

import asyncio
import time
import statistics
import sys
import os
import glob
import random

import aiohttp

SERVER = "http://localhost:8000"
AUDIO_DIR = "/home/shadeform/benchmark/data/mls_french"
AUDIO_FILES = sorted(glob.glob(f"{AUDIO_DIR}/mls_*.wav"))


async def send_request(session: aiohttp.ClientSession) -> dict:
    """Send a single transcribe request with a random audio file."""
    audio_file = random.choice(AUDIO_FILES)
    t0 = time.perf_counter()
    data = aiohttp.FormData()
    data.add_field("file", open(audio_file, "rb"), filename=os.path.basename(audio_file))
    try:
        async with session.post(f"{SERVER}/transcribe", data=data) as resp:
            elapsed = time.perf_counter() - t0
            if resp.status == 200:
                body = await resp.json()
                return {"ok": True, "latency": elapsed, "server_ms": body["processing_time_ms"]}
            else:
                return {"ok": False, "latency": elapsed, "error": resp.status}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"ok": False, "latency": elapsed, "error": str(e)}


async def test_sequential(n: int = 50):
    """Send n requests one after another, measure average latency."""
    print(f"\n{'='*60}")
    print(f"TEST 1: {n} sequential requests (random files from {len(AUDIO_FILES)} audios)")
    print(f"{'='*60}")

    latencies = []
    server_times = []
    async with aiohttp.ClientSession() as session:
        for i in range(n):
            result = await send_request(session)
            if result["ok"]:
                latencies.append(result["latency"])
                server_times.append(result["server_ms"])
                print(f"  [{i+1:3d}/{n}] {result['latency']*1000:7.1f}ms total, {result['server_ms']:7.1f}ms server")
            else:
                print(f"  [{i+1:3d}/{n}] FAILED: {result['error']}")

    if latencies:
        print(f"\nResults ({len(latencies)} successful / {n} total):")
        print(f"  Average total latency:  {statistics.mean(latencies)*1000:.1f}ms")
        print(f"  Median total latency:   {statistics.median(latencies)*1000:.1f}ms")
        print(f"  Min total latency:      {min(latencies)*1000:.1f}ms")
        print(f"  Max total latency:      {max(latencies)*1000:.1f}ms")
        print(f"  P95 total latency:      {sorted(latencies)[int(len(latencies)*0.95)]*1000:.1f}ms")
        print(f"  Average server time:    {statistics.mean(server_times):.1f}ms")


async def test_concurrency(rps: int, duration: int = 10):
    """Send `rps` requests per second for `duration` seconds."""
    interval = 1.0 / rps
    total_requests = rps * duration

    print(f"\n--- {rps} req/s for {duration}s ({total_requests} total) ---")

    tasks = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        t_start = time.perf_counter()
        for i in range(total_requests):
            target_time = t_start + i * interval
            now = time.perf_counter()
            if target_time > now:
                await asyncio.sleep(target_time - now)
            tasks.append(asyncio.create_task(send_request(session)))

        results = await asyncio.gather(*tasks)

    latencies = []
    errors = 0
    for r in results:
        if r["ok"]:
            latencies.append(r["latency"])
        else:
            errors += 1

    if latencies:
        avg = statistics.mean(latencies)
        med = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        mx = max(latencies)
        print(f"  OK: {len(latencies)}, Errors: {errors}")
        print(f"  Avg: {avg*1000:.0f}ms | Med: {med*1000:.0f}ms | P95: {p95*1000:.0f}ms | Max: {mx*1000:.0f}ms")
        return avg
    else:
        print(f"  All {errors} requests failed!")
        return float("inf")


async def test_ramp():
    """Ramp up concurrency until average latency exceeds 1 second."""
    print(f"\n{'='*60}")
    print("TEST 2: Ramp concurrency to find 1-second latency threshold")
    print(f"{'='*60}")

    rates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
    results = {}

    for rps in rates:
        avg = await test_concurrency(rps, duration=10)
        results[rps] = avg
        if avg > 1.0:
            print(f"\n  >>> Average latency exceeds 1s at {rps} req/s ({avg*1000:.0f}ms)")
            break

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'RPS':>5}  {'Avg Latency':>12}")
    print(f"  {'---':>5}  {'----------':>12}")
    for rps, avg in results.items():
        marker = " <<<" if avg > 1.0 else ""
        print(f"  {rps:>5}  {avg*1000:>10.0f}ms{marker}")


async def main():
    # Check server health
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{SERVER}/health") as resp:
                health = await resp.json()
                print(f"Server healthy: {health}")
        except Exception as e:
            print(f"Server not reachable: {e}")
            sys.exit(1)

    await test_sequential(50)
    await test_ramp()


if __name__ == "__main__":
    asyncio.run(main())
