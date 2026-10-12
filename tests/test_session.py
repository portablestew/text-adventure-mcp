"""Tests for session.py. Real TCP to tests/fakes.py; timing uses real time with wide tolerance (D4).

Some tests read Session._links and _locks directly. They are the only way to wait for the reader
task to notice a close, and they never assert on the registry's contents otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable

import pytest

from tests.fakes import StreamService, TurnService, wait_until
from text_adventure_mcp.session import (
    NOT_SENT_REASON,
    Address,
    Failure,
    Outcome,
    Session,
    parse_address,
)
from text_adventure_mcp.wire import Link, normalize_command

pytestmark = pytest.mark.anyio

Schedule = list[tuple[float, bytes]]


def echo(line: bytes) -> Schedule:
    return [(0.0, b"ok:" + line + b"\r\n")]


def connected(address: str) -> str:
    return f"[status] connected to {address}"


def lost(address: str) -> str:
    return f"[status] connection to {address} lost"


@contextlib.asynccontextmanager
async def managed_session() -> AsyncIterator[Session]:
    session = Session()
    try:
        yield session
    finally:
        await session.aclose()


async def send(
    session: Session,
    address: str,
    command: str,
    *,
    settle: float | None = None,
    timeout: float = 5.0,
) -> Outcome:
    return await session.exchange(
        parse_address(address),
        normalize_command(command),
        settle=settle,
        timeout=timeout,
    )


# --- Address parsing (E7, D5) -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "host", "port"),
    [("127.0.0.1:3000", "127.0.0.1", 3000), ("localhost:1", "localhost", 1), ("h:65535", "h", 65535)],
)
def test_parse_address_accepts(text: str, host: str, port: int) -> None:
    assert parse_address(text) == Address(host=host, port=port, text=text)


@pytest.mark.parametrize("text", ["nohost", ":3000", "h:", "h:0", "h:65536", "h:99999", "h:x", "h:-1"])
def test_parse_address_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_address(text)


# --- Connection (C1, C2, C3, C5) --------------------------------------------------------------


async def test_first_call_connects_then_reuses() -> None:
    async with TurnService(echo) as svc, managed_session() as session:
        first = await send(session, svc.address, "look", settle=None)
        second = await send(session, svc.address, "north")

        assert first == Outcome([connected(svc.address)], "ok:look\r\n")
        assert second == Outcome([], "ok:north\r\n")
        assert svc.accepts == 1


async def test_key_is_exact_string() -> None:
    async with TurnService(echo) as svc, managed_session() as session:
        await send(session, f"127.0.0.1:{svc.port}", "a")
        await send(session, f"localhost:{svc.port}", "b")

        assert svc.accepts == 2


async def test_no_idle_expiry() -> None:
    async with TurnService(echo) as svc, managed_session() as session:
        await send(session, svc.address, "a")
        await asyncio.sleep(1.5)
        again = await send(session, svc.address, "b")

        assert again == Outcome([], "ok:b\r\n")
        assert svc.accepts == 1


async def test_addresses_independent() -> None:
    async with (
        TurnService(lambda line: [(1.0, b"slow\r\n")]) as slow_svc,
        TurnService(echo) as fast_svc,
        managed_session() as session,
    ):
        slow = asyncio.create_task(send(session, slow_svc.address, "x"))
        fast = await send(session, fast_svc.address, "y")

        assert fast.text == "ok:y\r\n"
        assert not slow.done()
        await slow


# --- Receiving (R1, R8) -----------------------------------------------------------------------


async def test_between_calls_is_returned_by_next_call() -> None:
    async with StreamService() as svc, managed_session() as session:
        await send(session, svc.address, "", settle=0.0)
        await wait_until(lambda: svc.received == b"\r\n")

        svc.write(b"unsolicited")
        await wait_until(lambda: bytes(session._links[svc.address].raw) == b"unsolicited")
        result = await send(session, svc.address, "look", settle=0.0)

        assert result == Outcome([], "unsolicited")
        await wait_until(lambda: svc.received.endswith(b"look\r\n"))


async def test_serialized_calls_do_not_interleave() -> None:
    def slow_reply(line: bytes) -> Schedule:
        return [(0.3, b"re:" + line + b"\r\n")]

    async with TurnService(slow_reply) as svc, managed_session() as session:
        first = asyncio.create_task(send(session, svc.address, "first"))
        second = asyncio.create_task(send(session, svc.address, "second"))
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result.text == "re:first\r\n"
        assert second_result.text == "re:second\r\n"
        assert svc.received == b"first\r\nsecond\r\n"


async def test_lock_wait_timeout_leaves_buffer_for_holder() -> None:
    async with StreamService() as svc, managed_session() as session:
        address = svc.address
        await send(session, address, "", settle=0.0)

        holder = asyncio.create_task(send(session, address, "", settle=0.8, timeout=5.0))
        await wait_until(lambda: session._locks[address].locked())
        svc.write(b"hello")

        waiter = await send(session, address, "", settle=0.0, timeout=0.3)

        assert waiter == Outcome(["[status] timed out after 0.3s"], "")
        assert not holder.done()
        assert (await holder).text == "hello"


async def test_blocked_send_is_bounded_by_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # A write that never completes, as when the peer stops reading. The send is stubbed because a
    # real blocked write could not be produced on the test machine: its loopback accepted 1 GB
    # with no reader, so drain() never waited. The call must still return at its deadline, and
    # the socket must be dropped rather than reused.
    async def stuck_send(self: Link, payload: bytes) -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(Link, "send", stuck_send)
    async with TurnService(echo) as svc, managed_session() as session:
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await send(session, svc.address, "look", timeout=0.3)
        elapsed = loop.time() - started

        assert result == Outcome([connected(svc.address), "[status] timed out after 0.3s"], "")
        assert elapsed < 1.5
        assert svc.address not in session._links


async def test_connect_timeout_returns_timed_out_with_no_text(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_open(host: str, port: int, on_close: Callable[[Link], None]) -> Link:
        await asyncio.sleep(5)
        raise AssertionError("unreachable: the call should have timed out")

    monkeypatch.setattr(Link, "open", slow_open)
    async with managed_session() as session:
        result = await send(session, "127.0.0.1:9", "look", settle=None, timeout=0.3)

    assert result == Outcome(["[status] timed out after 0.3s"], "")


# --- Connection failures (C4, E8, A5, A7) -----------------------------------------------------


async def test_idle_close_with_nothing_retained_reconnects_silently() -> None:
    async with TurnService(echo) as svc, managed_session() as session:
        await send(session, svc.address, "a")
        svc.close()
        await wait_until(lambda: svc.address not in session._links)

        again = await send(session, svc.address, "b")

        assert again == Outcome([connected(svc.address)], "ok:b\r\n")
        assert svc.accepts == 2


async def test_idle_close_with_retained_bytes_reports_lost_and_does_not_send() -> None:
    async with StreamService() as svc, managed_session() as session:
        address = svc.address
        await send(session, address, "", settle=0.0)
        await wait_until(lambda: svc.received == b"\r\n")

        svc.write(b"unsolicited\r\n")
        svc.close()
        await wait_until(lambda: address in session._links and session._links[address].closed)
        received_before = bytes(svc.received)

        with pytest.raises(Failure) as info:
            await send(session, address, "look")
        failure = info.value
        assert failure.status == [lost(address)]
        assert failure.text == "unsolicited\r\n"
        assert failure.reason == NOT_SENT_REASON
        assert bytes(svc.received) == received_before

        after = await send(session, address, "", settle=0.0)
        assert after == Outcome([connected(address)], "")
        assert svc.accepts == 2


async def test_loss_mid_call_returns_partial_text() -> None:
    async with StreamService() as svc, managed_session() as session:
        address = svc.address
        await send(session, address, "", settle=0.0)
        await wait_until(lambda: svc.received == b"\r\n")

        call = asyncio.create_task(send(session, address, "look", settle=5.0, timeout=5.0))
        await wait_until(lambda: svc.received == b"\r\nlook\r\n")
        # Give the call time to reach its collection window, so this is a loss mid-call.
        await asyncio.sleep(0.1)
        svc.write(b"abc")
        svc.close()

        with pytest.raises(Failure) as info:
            await call
        assert info.value.status == [lost(address)]
        assert info.value.text == "abc"
        assert info.value.reason == ""
        assert address not in session._links


async def test_connect_refused_has_no_status_line(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real refused connect takes about 2 s on this Windows machine. The server tests cover the
    # real path; here an immediate ConnectionRefusedError exercises the same branch.
    async def refused_open(host: str, port: int, on_close: Callable[[Link], None]) -> Link:
        raise ConnectionRefusedError(10061, "refused")

    monkeypatch.setattr(Link, "open", refused_open)
    async with managed_session() as session:
        with pytest.raises(Failure) as info:
            await send(session, "127.0.0.1:9", "look")

    assert info.value.status == []
    assert info.value.text == ""
    assert info.value.reason.startswith("could not connect to 127.0.0.1:9: ")


async def test_unusable_host_is_a_connect_failure() -> None:
    # asyncio raises ValueError, not OSError, for a host it cannot resolve at all.
    async with managed_session() as session:
        with pytest.raises(Failure) as info:
            await send(session, "bad\x00host:80", "look")

    assert info.value.status == []
    assert info.value.reason.startswith("could not connect to bad\x00host:80: ")
