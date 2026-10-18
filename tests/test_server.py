"""Tests for server.py: the text_adventure tool, called in process through MCPServer.

Protocol-level details (isError flag on the wire, stdout hygiene) are covered by the stdio e2e test.
Timing uses real time with wide tolerance (D4).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tests.fakes import StreamService, TurnService, wait_until
from text_adventure_mcp.server import create_server, render_failure, render_outcome
from text_adventure_mcp.session import Failure, Outcome, Session

pytestmark = pytest.mark.anyio

Schedule = list[tuple[float, bytes]]

# The SDK prefixes every deliberate tool error with this. Our status and partial text follow it.
SDK_ERROR_PREFIX = "Error executing tool text_adventure: "


def echo(line: bytes) -> Schedule:
    return [(0.0, b"ok:" + line + b"\r\n")]


def silent(line: bytes) -> Schedule:
    return []


def connected(address: str) -> str:
    return f"[status] connected to {address}\n"


async def call(app: MCPServer, **arguments: Any) -> str:
    result = await app.call_tool("text_adventure", arguments)
    assert result.is_error is not True
    return result.content[0].text


async def call_error(app: MCPServer, **arguments: Any) -> str:
    with pytest.raises(ToolError) as info:
        await app.call_tool("text_adventure", arguments)
    return str(info.value)


def stub_exchange(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float | None, float]]:
    """Replace Session.exchange so argument defaults can be checked without a service."""
    seen: list[tuple[float | None, float]] = []

    async def fake_exchange(
        self: Session, address: object, payload: bytes, *, settle: float | None, timeout: float
    ) -> Outcome:
        seen.append((settle, timeout))
        return Outcome([], "")

    monkeypatch.setattr(Session, "exchange", fake_exchange)
    return seen


# --- Tool surface (T1-T6, N3) -----------------------------------------------------------------


async def test_exactly_one_tool_with_the_spec_parameters() -> None:
    app = create_server()
    tools = await app.list_tools()

    assert [tool.name for tool in tools] == ["text_adventure"]
    schema = tools[0].input_schema
    assert schema["required"] == ["command"]
    assert schema["properties"]["address"]["default"] == "127.0.0.1:3000"
    for name in ("settle_ms", "timeout_ms"):
        assert schema["properties"][name]["default"] is None
        kinds = {option.get("type") for option in schema["properties"][name]["anyOf"]}
        assert kinds == {"integer", "null"}


async def test_no_structured_output() -> None:
    app = create_server()
    tools = await app.list_tools()

    assert tools[0].output_schema is None


async def test_null_equals_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_exchange(monkeypatch)
    app = create_server()

    await call(app, command="look")
    await call(app, command="look", settle_ms=None, timeout_ms=None)

    assert seen == [(None, 20.0), (None, 20.0)]


@pytest.mark.parametrize(
    ("arguments", "expected_timeout"),
    [({}, 20.0), ({"settle_ms": 5000}, 25.0), ({"settle_ms": 0}, 20.0), ({"timeout_ms": 3000}, 3.0)],
)
async def test_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, Any],
    expected_timeout: float,
) -> None:
    seen = stub_exchange(monkeypatch)
    app = create_server()

    await call(app, command="look", **arguments)

    assert seen[-1][1] == expected_timeout


@pytest.mark.parametrize(
    ("name", "value"),
    [("settle_ms", -1), ("timeout_ms", -1), ("settle_ms", 300_001), ("timeout_ms", 300_001)],
)
async def test_out_of_range_is_rejected(name: str, value: int) -> None:
    app = create_server()

    message = await call_error(app, command="look", **{name: value})

    assert name in message
    assert "must be a whole number of milliseconds from 0 to 300000" in message


async def test_zero_timeout_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = stub_exchange(monkeypatch)
    app = create_server()

    await call(app, command="look", timeout_ms=0)

    assert seen[-1][1] == 0


async def test_hard_cap_wins_over_settle() -> None:
    async with TurnService(silent) as svc:
        app = create_server()
        started = time.monotonic()
        text = await call(app, command="look", address=svc.address, settle_ms=1000, timeout_ms=200)
        elapsed = time.monotonic() - started

    assert text == connected(svc.address) + "[status] timed out after 0.2s\n"
    assert elapsed < 0.9


# --- Address validation (E7, D5) --------------------------------------------------------------


@pytest.mark.parametrize("address", ["nohost", "h:0", "h:99999", "h:x"])
async def test_bad_address_is_a_tool_error(address: str) -> None:
    app = create_server()

    message = await call_error(app, command="look", address=address)

    assert "must be" in message


# --- Status text and receiving (ST1, ST4, ST6, R5, R9, R10) -----------------------------------


async def test_banner_comes_back_with_first_call_under_connected_line() -> None:
    async with TurnService(echo, banner=b"Welcome\r\n") as svc:
        app = create_server()

        text = await call(app, command="look", address=svc.address)

    assert text == connected(svc.address) + "Welcome\r\nok:look\r\n"


async def test_routine_reply_has_no_status_line() -> None:
    async with TurnService(echo) as svc:
        app = create_server()
        await call(app, command="look", address=svc.address)

        text = await call(app, command="north", address=svc.address)

    assert text == "ok:north\r\n"


async def test_silent_service_reports_timeout() -> None:
    async with TurnService(silent) as svc:
        app = create_server()
        started = time.monotonic()
        text = await call(app, command="look", address=svc.address, timeout_ms=500)
        elapsed = time.monotonic() - started

    assert text == connected(svc.address) + "[status] timed out after 0.5s\n"
    assert 0.4 <= elapsed < 1.5


async def test_partial_output_is_returned_on_timeout() -> None:
    async with StreamService() as svc:
        app = create_server()
        pending = asyncio.create_task(call(app, command="", address=svc.address, settle_ms=5000, timeout_ms=1000))
        await wait_until(lambda: svc.accepts == 1)
        svc.write(b"abc")

        text = await pending

    assert text == connected(svc.address) + "[status] timed out after 1s\nabc"


async def test_large_response_is_returned_whole() -> None:
    body = b"x" * 1_000_000 + b"\r\n"
    async with TurnService(lambda line: [(0.0, body)]) as svc:
        app = create_server()

        text = await call(app, command="look", address=svc.address, settle_ms=1500)

    assert text == connected(svc.address) + body.decode()


# --- Errors (ST2, ST5, ST7, E2, E3, E6) -------------------------------------------------------


async def test_connection_lost_mid_call_returns_status_and_partial_text() -> None:
    async with StreamService() as svc:
        app = create_server()
        await call(app, command="", address=svc.address, settle_ms=0)
        await wait_until(lambda: svc.received == b"\r\n")

        pending = asyncio.create_task(call_error(app, command="look", address=svc.address, settle_ms=5000))
        await wait_until(lambda: svc.received == b"\r\nlook\r\n")
        await asyncio.sleep(0.1)
        svc.write(b"abc")
        svc.close()
        message = await pending

    assert message == SDK_ERROR_PREFIX + f"[status] connection to {svc.address} lost\nabc"


async def test_connect_refused_is_a_tool_error_without_status() -> None:
    async with TurnService(echo) as svc:
        port = svc.port
    app = create_server()

    message = await call_error(app, command="look", address=f"127.0.0.1:{port}")

    assert message.startswith(SDK_ERROR_PREFIX + f"could not connect to 127.0.0.1:{port}: ")
    assert "[status]" not in message


def test_render_failure_puts_status_then_text_then_reason() -> None:
    failure = Failure(["[status] connection to a lost"], "partial", "Not sent: resend it.")

    assert render_failure(failure) == "[status] connection to a lost\npartial\nNot sent: resend it."


def test_render_outcome_puts_status_before_text() -> None:
    assert render_outcome(Outcome(["[status] connected to a"], "hi")) == "[status] connected to a\nhi"
