"""FastMCP app (MCPServer in mcp 2.x) and the text_adventure tool.

Thin layer: validates arguments, runs Session.exchange, and renders the result as plain text.
Only this module knows about MCP.
"""

from __future__ import annotations

import logging
import math
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from text_adventure_mcp import __version__
from text_adventure_mcp.session import Failure, Outcome, Session, parse_address
from text_adventure_mcp.wire import normalize_command

DEFAULT_ADDRESS = "127.0.0.1:3000"
DEFAULT_TIMEOUT_S = 20.0

TOOL_DESCRIPTION = """\
Send one line of text to a line-oriented TCP service and return what it sends back.

The connection stays open between calls, so the service keeps its state. Every call is a turn: an
empty command sends a bare newline. Without settle, the call returns when the reply goes quiet for
200 ms, or 1 s after the first byte, whichever comes first. With settle, the call collects output for
exactly that many seconds. timeout caps the whole call and defaults to 20 s, or 20 s plus settle.

Output starts with [status] lines only when a connection is opened, a connection is lost, or a call
times out. Anything else is the service's own text.\
"""

log = logging.getLogger(__name__)


def create_server() -> MCPServer:
    """Build the MCP app. Each server gets its own Session, so tests start with clean state."""
    app = MCPServer("text-adventure-mcp", version=__version__)
    session = Session()

    @app.tool(structured_output=False, description=TOOL_DESCRIPTION)
    async def text_adventure(
        command: str,
        address: str = DEFAULT_ADDRESS,
        settle: float | None = None,
        timeout: float | None = None,
    ) -> str:
        _check_seconds("settle", settle)
        _check_seconds("timeout", timeout)
        try:
            target = parse_address(address)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        if timeout is None:
            timeout = DEFAULT_TIMEOUT_S if settle is None else DEFAULT_TIMEOUT_S + settle

        try:
            outcome = await session.exchange(
                target,
                normalize_command(command),
                settle=settle,
                timeout=timeout,
            )
        except Failure as failure:
            log.info("call to %s failed: %s", target.text, failure.reason or failure.status)
            raise ToolError(render_failure(failure)) from failure
        return render_outcome(outcome)

    return app


def render_outcome(outcome: Outcome) -> str:
    """Status lines first, then the service text exactly as received."""
    return _status_block(outcome.status) + outcome.text


def render_failure(failure: Failure) -> str:
    """Status lines, then the partial text, then the reason, if there is one."""
    text = _status_block(failure.status) + failure.text
    if failure.reason:
        if text and not text.endswith("\n"):
            text += "\n"
        text += failure.reason
    return text


def _status_block(lines: list[str]) -> str:
    return "".join(line + "\n" for line in lines)


def _check_seconds(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ToolError(f"{name} must be a finite number >= 0 (got {value})")


def main() -> None:
    # stdout is the protocol channel on stdio, so logs go to stderr only.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    create_server().run()


if __name__ == "__main__":
    main()
