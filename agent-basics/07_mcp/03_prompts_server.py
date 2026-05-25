"""MCP prompts — parameterised templates the server ships.

A prompt is rendered by the client into a chat message before the model
runs. Tools = "DO this", prompts = "ASK about this AS IF the user said it."

Why prompts (and not just include it in the system prompt): they let an
MCP server ship its own prompt engineering. The agent calls
`/prompts/audit_voice_quality voice_id=...` without knowing the internal
phrasing, and the server can update the template without touching every
consumer.
"""

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("prompts-demo")


@mcp.prompt()
def summarise(text: str, max_words: int = 50) -> str:
    """Simple string return -> renders as one user message."""
    return (
        f"Please produce a summary of the following text in no more "
        f"than {max_words} words. Output ONLY the summary, no preamble.\n\n"
        f"<text>\n{text}\n</text>"
    )


@mcp.prompt()
def audit_voice_quality(transcript: str, target_voice_id: str) -> list[base.Message]:
    """Multi-turn: return a list of messages to pre-roll the conversation.

    Useful for system+user pairs and few-shot exemplars.
    """
    return [
        base.AssistantMessage(
            "I'm a voice QA specialist for ElevenLabs. I review dubbing "
            "for clarity, pronunciation, and voice consistency.",
        ),
        base.UserMessage(
            f"Audit the following transcript for voice issues.\n\n"
            f"Target voice ID: {target_voice_id}\n"
            f"Transcript:\n{transcript}\n\n"
            f"Return: (a) pass/fail, (b) per-line issues with timestamps, "
            f"(c) recommended re-record list."
        ),
    ]


@mcp.prompt()
def write_dubbing_brief(
    project_name: str, target_language: str, style: str = "natural"
) -> str:
    """Embeds service-side conventions into the prompt — consumers don't
    need to learn them separately. This is the "ship your prompt eng" pattern.
    """
    return f"""Write a dubbing brief for project "{project_name}".

Target language: {target_language}
Style: {style}

Follow these conventions:
  - Specify exact target voice IDs, never just descriptive names.
  - Call out abbreviation handling (Dr., Mrs., etc.) explicitly.
  - For numerical content (dates, prices), state spell-out vs digit reading.
  - Note any code-switched terms that should NOT be translated.
  - Set max-segment-length to 200 chars to align with Flash v2.5's prosody window.
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
