"""ElevenLabs practical-round worked example: streaming TTS chunker.

Problem: a voice agent receives streaming LLM tokens and must send chunks
to TTS. Shorter chunks = lower first-audio latency; sentence-aligned
chunks = better prosody. Design the chunker that balances these.

Approach: priority-ordered flush rules.
  1. Sentence boundary (. ! ?) -> flush. Best prosody.
  2. Clause boundary (, ; :) past soft_size -> flush. Bounded latency.
  3. Hard max_size reached -> force flush. Hard cap.

Plus: handle abbreviations ("Dr.", "Mr.") so we don't flush mid-sentence.
Plus: a tick() called periodically to flush after long silences (LLM
done generating but no terminator).
"""

import asyncio
import time

SENTENCE_END = {".", "!", "?"}
CLAUSE_END = {",", ";", ":"}
ABBREVIATIONS = {"mr.", "mrs.", "dr.", "ms.", "st.", "jr.", "sr."}


class StreamingChunker:
    def __init__(self, on_chunk, *, max_size=200, soft_size=80, max_wait_s=0.5):
        self.on_chunk = on_chunk
        self.max_size = max_size
        self.soft_size = soft_size
        self.max_wait_s = max_wait_s
        self.buffer = []
        self.last_token_time = time.time()

    def add_token(self, token):
        if not token:
            return
        self.buffer.append(token)
        self.last_token_time = time.time()
        text = " ".join(self.buffer)

        # Priority order matters: a sentence boundary always beats a clause one.
        if (
            self._sentence_boundary(text)
            or len(text) > self.soft_size
            and text[-1] in CLAUSE_END
            or len(text) >= self.max_size
        ):
            self._flush()

    def tick(self):
        """Called periodically. Flush stale buffers (LLM stopped without punctuation)."""
        if self.buffer and time.time() - self.last_token_time >= self.max_wait_s:
            self._flush()

    def end(self):
        """Stream done — flush whatever's left."""
        if self.buffer:
            self._flush()

    def _sentence_boundary(self, text):
        if not text or text[-1] not in SENTENCE_END:
            return False
        # Don't flush on "Dr.", "Mrs.", etc.
        return text.split()[-1].lower() not in ABBREVIATIONS

    def _flush(self):
        text = " ".join(self.buffer)
        self.buffer = []
        self.on_chunk(text)


# --- tests ------------------------------------------------------------------


def test_sentence_boundary():
    chunks = []
    c = StreamingChunker(chunks.append)
    for tok in ["Hello", "world.", "How", "are", "you?"]:
        c.add_token(tok)
    assert chunks == ["Hello world.", "How are you?"], chunks
    print("sentence boundary OK")


def test_abbreviation_does_not_flush():
    chunks = []
    c = StreamingChunker(chunks.append)
    for tok in ["Hello", "Dr.", "Smith.", "How", "are", "you?"]:
        c.add_token(tok)
    assert chunks == ["Hello Dr. Smith.", "How are you?"], chunks
    print("abbreviation handling OK")


def test_clause_flush_past_soft_size():
    chunks = []
    c = StreamingChunker(chunks.append, soft_size=40)
    text = (
        "This is a very long sentence with many words "
        "exceeding the soft limit, but no period yet."
    )
    for tok in text.split():
        c.add_token(tok)
    assert any(s.endswith(",") for s in chunks), chunks
    print("clause flush past soft size OK")


def test_force_flush_at_max():
    chunks = []
    c = StreamingChunker(chunks.append, max_size=50)
    for _ in range(30):
        c.add_token("word")
    assert chunks, "should have flushed"
    print("force flush at max size OK")


def test_end_flushes_remainder():
    chunks = []
    c = StreamingChunker(chunks.append)
    c.add_token("Trailing")
    c.add_token("text")
    c.end()
    assert chunks == ["Trailing text"]
    print("end() flushes remainder OK")


# --- async demo: tick() integrates with the event loop ----------------------


async def demo_async():
    chunks = []
    c = StreamingChunker(chunks.append, soft_size=30, max_wait_s=0.3)

    async def emit():
        for tok in ["Hello,", "world.", "How", "are", "you", "doing", "today?"]:
            c.add_token(tok)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.5)  # long pause -> tick() should flush
        for tok in ["Final", "thought", "without", "a", "period"]:
            c.add_token(tok)
            await asyncio.sleep(0.02)
        c.end()

    async def ticker():
        while True:
            await asyncio.sleep(0.1)
            c.tick()

    emit_task = asyncio.create_task(emit())
    tick_task = asyncio.create_task(ticker())
    await emit_task
    tick_task.cancel()
    try:
        await tick_task
    except asyncio.CancelledError:
        pass

    print("\nasync streaming flushed:")
    for ch in chunks:
        print(f"  -> {ch!r}")


async def main():
    test_sentence_boundary()
    test_abbreviation_does_not_flush()
    test_clause_flush_past_soft_size()
    test_force_flush_at_max()
    test_end_flushes_remainder()
    await demo_async()


if __name__ == "__main__":
    asyncio.run(main())
