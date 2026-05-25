"""Idempotency keys: make retries safe for side-effecting operations.

When a tool call (send email, charge card, create issue) retries after
a network blip, you do NOT want the side effect twice. An idempotency
key lets the server de-dup: if it's seen this key before, it returns
the prior result instead of repeating the action.

Two strategies for the key:
  1. Deterministic hash of the request inputs (this script).
     Pros: no client-side state, same inputs always collapse.
     Cons: legitimate distinct requests with identical inputs collide.
  2. Explicit client-provided UUID per operation.
     Pros: distinguishes "same payload, different intent."
     Cons: caller must remember and pass it across retries.

Real APIs (Stripe etc.) use a header `Idempotency-Key`. This demo
simulates a server-side dedup table.
"""

import asyncio
import hashlib
import json
from typing import Any


class FakeEmailServer:
    """Tracks seen idempotency keys and returns prior results on replay."""

    def __init__(self) -> None:
        self.seen: dict[str, dict[str, Any]] = {}
        self.send_count = 0  # true side-effect counter

    async def send(self, payload: dict, idempotency_key: str) -> dict:
        if idempotency_key in self.seen:
            return {**self.seen[idempotency_key], "replayed": True}
        self.send_count += 1  # only increments on a real send
        result = {"message_id": f"msg-{self.send_count}", "payload": payload}
        self.seen[idempotency_key] = result
        return {**result, "replayed": False}


def deterministic_key(payload: dict) -> str:
    """Hash the canonical JSON of the payload. Stable across retries."""
    canonical = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


async def send_email_idempotent(server: FakeEmailServer, payload: dict) -> dict:
    return await server.send(payload, idempotency_key=deterministic_key(payload))


async def main() -> None:
    server = FakeEmailServer()
    payload = {"to": "alice@example.com", "subject": "Hello", "body": "hi"}

    # Simulate three retries of the same logical send (e.g., network blips)
    for i in range(3):
        result = await send_email_idempotent(server, payload)
        print(f"retry {i}: {result}")

    # Distinct payload -> distinct key -> real second send
    other = {"to": "bob@example.com", "subject": "Hello", "body": "hi"}
    result = await send_email_idempotent(server, other)
    print(f"other:   {result}")

    print(f"\ntotal real sends: {server.send_count} (should be 2, not 4)")


if __name__ == "__main__":
    asyncio.run(main())
