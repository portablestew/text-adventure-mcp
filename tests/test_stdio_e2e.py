"""End to end over stdio: the real server in a subprocess, driven by the SDK's stdio client.

The fake services run in this process on 127.0.0.1. Set TAM_WRITE_TRANSCRIPT to a file path to save
the requests and results as text (that is how docs/4-e2e.txt was produced). Without it, the test
has no side effects.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from tests.fakes import Schedule, TurnService

pytestmark = pytest.mark.anyio

GAME_BANNER = b"Welcome to the test game.\r\n> "


def game_reply(line: bytes) -> Schedule:
    """Game A: look shows colour codes around the room name; partial sends 'abc' then goes silent."""
    if line == b"look":
        return [(0.0, b"\x1b[1mHall\x1b[0m\r\n")]
    if line == b"partial":
        return [(0.1, b"abc")]
    if line == b"a":
        return [(0.0, b"A-reply\r\n")]
    if line == b"b":
        return [(0.0, b"B-reply\r\n")]
    return []


def cellar_reply(line: bytes) -> Schedule:
    """Game B: a second address with its own state, so the interleaving is visible in the text."""
    if line == b"look":
        return [(0.0, b"Cellar\r\n")]
    return []


class Game(TurnService):
    """Like game_reply, but 'quit' sends a goodbye and drops the connection mid-call."""

    async def on_data(self, data: bytes) -> None:
        if b"quit" in data:
            self.write(b"bye\r\n")
            self.close()
            return
        await super().on_data(data)


def closed_port() -> int:
    """A port with nothing listening on it, so a connect is refused."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Transcript:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str) -> None:
        self.lines += ["", f"## {title}"]

    def note(self, text: str) -> None:
        self.lines.append(text)

    def save(self, path: str) -> None:
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


async def call(
    session: ClientSession, log: Transcript, label: str, **arguments: Any
) -> tuple[CallToolResult, str, float]:
    started = time.monotonic()
    result = await session.call_tool("text_adventure", arguments)
    elapsed = time.monotonic() - started
    text = "".join(block.text for block in result.content if hasattr(block, "text"))
    log.section(label)
    log.note(f"request:    tools/call text_adventure {json.dumps(arguments)}")
    log.note(
        f"result:     isError={bool(result.is_error)} "
        f"structuredContent={result.structured_content is not None} elapsed={elapsed:.2f}s"
    )
    log.note(f"text:       {text!r}")
    return result, text, elapsed


async def test_stdio_scenario() -> None:
    log = Transcript()
    async with Game(game_reply, GAME_BANNER) as game, Game(cellar_reply, b"Second game.\r\n> ") as cellar:
        address_a = game.address
        address_b = cellar.address
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "text_adventure_mcp.server"]
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()

            # T1, T2, A4, N3: one tool, the spec's parameters, plain text (no structured output).
            log.section("tools/list")
            listed = await session.list_tools()
            names = [tool.name for tool in listed.tools]
            log.note(f"tool names: {names}")
            assert names == ["text_adventure"]
            schema = listed.tools[0].input_schema
            log.note(f"inputSchema: {json.dumps(schema, sort_keys=True)}")
            assert schema["required"] == ["command"]
            assert schema["properties"]["address"]["default"] == "127.0.0.1:3000"
            for name in ("settle_ms", "timeout_ms"):
                assert schema["properties"][name]["default"] is None
                assert {"type": "null"} in schema["properties"][name]["anyOf"]

            # C3, ST1, R10: the first call connects and returns the banner under the connected line.
            _, text, _ = await call(session, log, "first call connects (empty command)", command="", address=address_a)
            assert text == f"[status] connected to {address_a}\n" + GAME_BANNER.decode()

            # S1, R3, CL1: a turn, with the escape codes stripped and no status line.
            _, text, _ = await call(session, log, "turn round trip with ANSI codes", command="look", address=address_a)
            assert text == "Hall\r\n"

            # R6, R7: settle_ms is a fixed window, so the call runs for the full second.
            result, text, elapsed = await call(
                session, log, "settle_ms=1000 fixed window", command="look", address=address_a, settle_ms=1000
            )
            assert text == "Hall\r\n"
            assert 0.9 <= elapsed < 2.5

            # R9, ST3, ST5: timeout returns the partial text under a timed-out line, not as an error.
            result, text, elapsed = await call(
                session, log, "timeout with partial text", command="partial", address=address_a, settle_ms=5000, timeout_ms=1000
            )
            assert result.is_error is not True
            assert text == "[status] timed out after 1s\nabc"
            assert 0.9 <= elapsed < 2.5

            # ERR: a malformed address is a tool error that names the problem.
            result, text, _ = await call(session, log, "bad address", command="look", address="nohost")
            assert result.is_error is True
            assert "address must be host:port" in text

            # C1, C2, C7: two addresses interleaved, each with its own state.
            for label, address, expected in [
                ("interleave: A", address_a, "Hall\r\n"),
                ("interleave: B (first call, connects)", address_b, None),
                ("interleave: A again", address_a, "Hall\r\n"),
                ("interleave: B again", address_b, "Cellar\r\n"),
            ]:
                _, text, _ = await call(session, log, label, command="look", address=address)
                if expected is None:
                    assert text.startswith(f"[status] connected to {address_b}\n")
                    assert text.endswith("Cellar\r\n")
                else:
                    assert text == expected
            assert game.accepts == 1 and cellar.accepts == 1

            # C6: two concurrent calls on one address are serialized; each gets only its own reply.
            started = time.monotonic()
            (_, first, _), (_, second, _) = await asyncio.gather(
                call(session, log, "concurrent calls on A: first", command="a", address=address_a, settle_ms=500),
                call(session, log, "concurrent calls on A: second", command="b", address=address_a, settle_ms=500),
            )
            combined = time.monotonic() - started
            log.section("concurrent calls on A: serialized")
            log.note(f"combined elapsed: {combined:.2f}s (two 0.5s windows back to back)")
            assert sorted([first, second]) == ["A-reply\r\n", "B-reply\r\n"]
            assert combined >= 0.9

            # C4, E3, A8, ST2: the service drops the socket mid-call. Partial text rides with the error.
            result, text, _ = await call(session, log, "connection lost mid-call", command="quit", address=address_a)
            assert result.is_error is True
            assert text == f"Error executing tool text_adventure: [status] connection to {address_a} lost\nbye\r\n"

            # C4, E1, C3: the next call reconnects and says so.
            _, text, _ = await call(session, log, "next call reconnects", command="look", address=address_a)
            assert text.startswith(f"[status] connected to {address_a}\n")
            assert text.endswith("Hall\r\n")
            assert game.accepts == 2

            # E2, ST7: nothing listening on the port is a tool error with no status line.
            port = closed_port()
            result, text, _ = await call(
                session, log, "connect refused", command="look", address=f"127.0.0.1:{port}"
            )
            assert result.is_error is True
            assert text.startswith(f"Error executing tool text_adventure: could not connect to 127.0.0.1:{port}: ")
            assert "[status]" not in text

    log.section("result")
    log.note("all stdio scenario assertions passed")

    transcript_path = os.environ.get("TAM_WRITE_TRANSCRIPT")
    if transcript_path:
        log.save(transcript_path)
