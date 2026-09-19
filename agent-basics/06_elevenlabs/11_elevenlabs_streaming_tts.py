"""Streaming TTS against the actual ElevenLabs API.

The prep doc is unambiguous: "Build something small with the ElevenLabs
API and ship it before the interview. This is the single highest-yield
prep activity." This script is the runnable artifact.

What this demonstrates:
  - AsyncElevenLabs client with streaming TTS — bytes arrive in chunks
    as the model generates, not after.
  - Time-to-first-byte measurement (the latency number that matters for
    a voice agent, NOT total synthesis time).
  - Concurrent synthesis across multiple texts with asyncio.gather —
    same pattern as 03_anthropic_agents/02_parallel_tool_calls.py but
    over a different API.
  - Writing the streamed audio to disk while streaming, so the file
    grows during the synthesis call (you can play it before it ends).

Requires: ELEVENLABS_API_KEY env var. Get one at https://elevenlabs.io
(free tier covers casual experimentation).

Default voice: 21m00Tcm4TlvDq8ikWAM (Rachel) from the public Voice Library.
Default model: eleven_flash_v2_5 — the same low-latency model the prep
doc names as the "75ms first-byte" target.
"""

import asyncio
import os
import time
from pathlib import Path

from elevenlabs.client import AsyncElevenLabs

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel — public preset
MODEL_ID = "eleven_flash_v2_5"


def get_client() -> AsyncElevenLabs:
    if not os.environ.get("ELEVENLABS_API_KEY"):
        raise SystemExit("Set ELEVENLABS_API_KEY to run this demo. Get one at https://elevenlabs.io")
    return AsyncElevenLabs()


async def stream_to_file(
    client: AsyncElevenLabs,
    text: str,
    out_path: Path,
) -> tuple[Path, float, int]:
    """Returns (file path, time-to-first-byte in seconds, total bytes written)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    first_byte_at: float | None = None
    total_bytes = 0

    stream = client.text_to_speech.stream(
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        text=text,
        output_format="mp3_44100_128",
    )

    with out_path.open("wb") as f:
        async for chunk in stream:
            if not chunk:
                continue
            if first_byte_at is None:
                first_byte_at = time.perf_counter() - t0
            f.write(chunk)
            total_bytes += len(chunk)

    assert first_byte_at is not None, "stream yielded no bytes"
    return out_path, first_byte_at, total_bytes


async def main() -> None:
    client = get_client()
    out_dir = Path("/tmp/elevenlabs_demo")

    # Demo 1: single utterance, time-to-first-byte measurement.
    print("--- single synthesis ---")
    path, ttfb, size = await stream_to_file(
        client,
        "Hello from the ElevenLabs streaming API. This text is being synthesised in real time.",
        out_dir / "single.mp3",
    )
    print(f"  wrote {size} bytes to {path}")
    print(f"  time-to-first-byte: {ttfb * 1000:.0f}ms")
    print("  (target per prep doc: ~75ms for Flash v2.5)")

    # Demo 2: three utterances synthesised concurrently.
    print("\n--- 3 concurrent syntheses (asyncio.gather) ---")
    texts = [
        "Welcome to ElevenLabs.",
        "This is the Flash v2.5 model running concurrently.",
        "Each utterance streams independently over its own request.",
    ]
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(stream_to_file(client, t, out_dir / f"concurrent_{i}.mp3") for i, t in enumerate(texts)),
    )
    elapsed = time.perf_counter() - t0
    for path, ttfb, size in results:
        print(f"  {path.name}: ttfb={ttfb * 1000:.0f}ms  bytes={size}")
    print(f"  total wall-clock: {elapsed:.2f}s (vs sum-of-ttfbs which would be sequential)")


if __name__ == "__main__":
    asyncio.run(main())
