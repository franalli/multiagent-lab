"""Real-world I/O: fetch many URLs sequentially vs concurrently.

This is where asyncio shines — most time is spent waiting on the network,
not the CPU. Compare wall-clock times.
"""

import asyncio
import time

import httpx

URLS = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
]


async def fetch(client: httpx.AsyncClient, url: str) -> int:
    r = await client.get(url, timeout=10)
    return r.status_code


async def sequential(client: httpx.AsyncClient) -> list[int]:
    return [await fetch(client, u) for u in URLS]


async def concurrent(client: httpx.AsyncClient) -> list[int]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(client, u)) for u in URLS]
    return [t.result() for t in tasks]


async def main() -> None:
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        codes = await sequential(client)
        print(f"sequential: {time.perf_counter() - t0:.2f}s, codes={codes}")

        t0 = time.perf_counter()
        codes = await concurrent(client)
        print(f"concurrent: {time.perf_counter() - t0:.2f}s, codes={codes}")


if __name__ == "__main__":
    asyncio.run(main())
