"""Real-world MCP server: ElevenLabs TTS wrapped as MCP tools.

Design choices to notice (the part that actually takes thought):
  - Granularity: split into `list_voices` + `synthesise` rather than one
    god-method — smaller tools compose better in Claude's chain-of-tool-calls.
  - Audio output: write to a temp file, return the path. MCP supports
    binary blocks, but a path is simpler for the agent to pass downstream.
  - Failures: let exceptions propagate; the SDK turns them into
    isError=True. No `try: ... except: return "error"` wrappers.
  - Lazy client construction so missing creds fail with a clear message
    at call time, not at import.

Requires: ELEVENLABS_API_KEY.
"""

import os
import tempfile
from pathlib import Path

from elevenlabs.client import ElevenLabs
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("elevenlabs-tts")

# Reasonable defaults. The MODEL is the same Flash v2.5 used elsewhere
# in the playground (06_elevenlabs/) — low-latency, good for streaming.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel" — public preset
DEFAULT_MODEL_ID = "eleven_flash_v2_5"


def _client() -> ElevenLabs:
    """Lazy SDK client. Missing creds fail with a clear error at call time."""
    if not os.environ.get("ELEVENLABS_API_KEY"):
        raise RuntimeError("ELEVENLABS_API_KEY not set; cannot synthesise.")
    return ElevenLabs()


@mcp.tool()
def synthesise(
    text: str,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    output_path: str | None = None,
) -> dict:
    """Synthesise `text` to an mp3 file. Returns {path, bytes, voice_id, model_id}.

    Sync function — MCP SDK runs sync tools in an executor automatically.
    """
    client = _client()
    audio_bytes = b"".join(
        client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            output_format="mp3_44100_128",
        )
    )
    # tempfile with delete=False so the file OUTLIVES this process and the
    # calling agent can pass the path onwards.
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp3", prefix="elevenlabs_")
        os.close(fd)
    Path(output_path).write_bytes(audio_bytes)
    return {
        "path": output_path,
        "bytes": len(audio_bytes),
        "voice_id": voice_id,
        "model_id": model_id,
    }


@mcp.tool()
def list_voices(limit: int = 10) -> list[dict]:
    """List voices. Projects to {voice_id, name, category} to keep payloads small."""
    client = _client()
    page = client.voices.search(page_size=limit)
    return [
        {
            "voice_id": v.voice_id,
            "name": v.name,
            "category": getattr(v, "category", None),
        }
        for v in (page.voices or [])
    ][:limit]


@mcp.resource("config://elevenlabs/usage-tips")
def usage_tips() -> str:
    """Reference text the agent consults — pure read, fits as a resource."""
    return """ElevenLabs TTS usage tips:

- Interactive / agent use: model_id="eleven_flash_v2_5" (75ms first-byte).
- Studio / long-form: model_id="eleven_multilingual_v2" (better prosody, slower).
- Provide voice_id, not voice name — names aren't unique across shared libraries.
- Handle "Dr." / "Mrs." in upstream chunking, not here.
- Multilingual: set voice_settings.language_code OR let the model auto-detect.
"""


if __name__ == "__main__":
    # For remote deployment, swap to transport="streamable-http" (file 05).
    mcp.run(transport="stdio")
