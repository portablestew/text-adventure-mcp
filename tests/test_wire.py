"""Tests for wire.py. Real TCP to tests/fakes.py.

Timing tests use real time. Assertions are ordering or wide bounds, never exact counts, so they
hold under Windows timer jitter. Each bound is chosen so a broken implementation fails it by a
large margin (for example, a missing cap would run for about 1.7 s; the cap test expects 1.0 s and
rejects anything above 1.4 s).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest

from tests.fakes import FakeService, wait_until
from text_adventure_mcp.wire import Link, LinkLost, decode, normalize_command

pytestmark = pytest.mark.anyio


def now() -> float:
    return asyncio.get_running_loop().time()


@contextlib.asynccontextmanager
async def open_link(svc: FakeService) -> AsyncIterator[tuple[Link, list[Link]]]:
    """Connect a Link to svc. Yields (link, closed_callbacks). Closes the link on exit."""
    closed: list[Link] = []
    link = await Link.open("127.0.0.1", svc.port, closed.append)
    await wait_until(lambda: svc.accepts > 0)
    try:
        yield link, closed
    finally:
        await link.aclose()


# --- Sending: normalization (S1, S2, S4, A11) -------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("", b"\r\n"),
        ("look", b"look\r\n"),
        ("a\nb", b"a\r\nb\r\n"),
        ("a\r\nb\r\n", b"a\r\nb\r\n"),
        ("a\r\n\n", b"a\r\n\r\n"),
        ("look\r", b"look\r\r\n"),
        ("a\rb", b"a\rb\r\n"),
        ("é", "é".encode() + b"\r\n"),
    ],
)
def test_normalize_table(command: str, expected: bytes) -> None:
    assert normalize_command(command) == expected


async def test_empty_command_reaches_service_as_crlf() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        await link.send(normalize_command(""))
        await wait_until(lambda: svc.received == b"\r\n")


async def test_multiline_command_is_one_payload() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        await link.send(normalize_command("a\nb"))
        await wait_until(lambda: svc.received == b"a\r\nb\r\n")


async def test_send_writes_once_per_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        writes: list[bytes] = []
        original = link._writer.write

        def spy(data: bytes) -> None:
            writes.append(data)
            original(data)

        monkeypatch.setattr(link._writer, "write", spy)
        await link.send(b"look\r\n")
        assert writes == [b"look\r\n"]


# --- Decoding (CL7, CL8, A13) ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "text", "tail"),
    [
        (b"abc", "abc", b""),
        ("é".encode(), "é", b""),
        (b"ab\xe4\xbd", "ab", b"\xe4\xbd"),
        (b"\xf0\x9f\x98", "", b"\xf0\x9f\x98"),
        (b"\xff", "�", b""),
        (b"ok\xffok", "ok�ok", b""),
        (b"\xa9", "�", b""),
    ],
)
def test_decode(data: bytes, text: str, tail: bytes) -> None:
    assert decode(data) == (text, tail)


# --- Link: buffer ownership and lifecycle ----------------------------------------------------


async def test_buffer_kept_across_idle_gap() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"hello")
        await wait_until(lambda: bytes(link.raw) == b"hello")
        assert link.has_retained()
        assert link.take() == "hello"
        assert not link.has_retained()


async def test_one_connection_for_many_sends() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        await link.send(b"one\r\n")
        await link.send(b"two\r\n")
        await wait_until(lambda: svc.received == b"one\r\ntwo\r\n")
        assert svc.accepts == 1


async def test_peer_close_marks_closed_and_fires_callback_once() -> None:
    async with FakeService() as svc, open_link(svc) as (link, closed):
        svc.close()
        await wait_until(lambda: link.closed)
        assert closed == [link]


async def test_send_after_peer_close_raises() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.close()
        await wait_until(lambda: link.closed)
        with pytest.raises(LinkLost):
            await link.send(b"x\r\n")


async def test_aclose_fires_callback_once() -> None:
    async with FakeService() as svc:
        closed: list[Link] = []
        link = await Link.open("127.0.0.1", svc.port, closed.append)
        await link.aclose()
        await link.aclose()
        assert link.closed
        assert closed == [link]


# --- Receive windows: turn mode (R2, R3, R4, R5, A2, A3) -------------------------------------


async def test_quiet_200ms_completes_response() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        # "late" is far enough after the first response that the first call cannot see it.
        svc.play([(0.0, b"ab"), (0.05, b"c"), (0.8, b"late")])
        start = now()
        timed_out = await link.collect(settle=None, timeout_at=start + 5)
        elapsed = now() - start
        assert not timed_out
        assert 0.15 <= elapsed <= 0.6
        assert link.take() == "abc"

        await link.collect(settle=None, timeout_at=now() + 5)
        assert link.take() == "late"


async def test_backlog_still_waits_quiet_period() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"old")
        await wait_until(lambda: bytes(link.raw) == b"old")
        start = now()
        timed_out = await link.collect(settle=None, timeout_at=start + 5)
        elapsed = now() - start
        assert not timed_out
        # Nothing new arrives, so only the quiet period can make the call wait this long.
        assert 0.15 <= elapsed <= 0.6
        assert link.take() == "old"


async def test_cap_1000ms_bounds_a_service_that_never_stops() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        # One byte every 20 ms for 1.5 s. Gaps stay far below the 200 ms quiet period, so the
        # call can only end at the 1000 ms cap. Without a cap it would run to about 1.7 s.
        total = 75
        svc.play([(0.0, b"x")] + [(0.02, b"x")] * (total - 1))
        start = now()
        timed_out = await link.collect(settle=None, timeout_at=start + 5)
        elapsed = now() - start
        assert not timed_out
        assert 0.9 <= elapsed <= 1.4
        first = link.take()
        assert 0 < len(first) < total

        # Whatever arrived after the cap is kept and comes out on later calls; nothing is lost.
        received = len(first)
        deadline = now() + 5
        while received < total:
            assert now() < deadline, "remaining bytes never arrived"
            await link.collect(settle=None, timeout_at=now() + 5)
            received += len(link.take())
        assert received == total


async def test_silent_service_times_out_with_no_text() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        start = now()
        timed_out = await link.collect(settle=None, timeout_at=start + 0.5)
        elapsed = now() - start
        assert timed_out
        assert 0.45 <= elapsed <= 0.9
        assert link.take() == ""


async def test_timeout_before_cap_returns_partial_text() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        # One byte every 20 ms for 2 s: quiet never triggers, and the 0.5 s timeout cuts the call.
        total = 100
        svc.play([(0.0, b"x")] + [(0.02, b"x")] * (total - 1))
        start = now()
        timed_out = await link.collect(settle=None, timeout_at=start + 0.5)
        elapsed = now() - start
        assert timed_out
        assert 0.45 <= elapsed <= 0.9
        assert 0 < len(link.take()) < total


async def test_prompt_is_not_special() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.play([(0.0, b"You see a door.\r\n> ")])
        start = now()
        await link.collect(settle=None, timeout_at=start + 5)
        elapsed = now() - start
        assert 0.15 <= elapsed <= 0.6
        assert link.take().endswith("> ")


# --- Receive windows: stream mode (R6, R7, R9) -----------------------------------------------


async def test_stream_window_is_fixed_not_debounced() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        # A 400 ms gap is twice the 200 ms quiet period. A debounced window would stop before "b".
        svc.play([(0.0, b"a"), (0.4, b"b")])
        start = now()
        timed_out = await link.collect(settle=0.8, timeout_at=start + 5)
        elapsed = now() - start
        assert not timed_out
        assert 0.75 <= elapsed <= 1.2
        assert link.take() == "ab"


async def test_stream_timeout_cuts_window_short() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.play([(0.0, b"a")])
        start = now()
        timed_out = await link.collect(settle=5.0, timeout_at=start + 0.6)
        elapsed = now() - start
        assert timed_out
        assert 0.5 <= elapsed <= 1.5
        assert link.take() == "a"


async def test_settle_zero_returns_backlog_immediately() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"x")
        await wait_until(lambda: bytes(link.raw) == b"x")
        start = now()
        timed_out = await link.collect(settle=0, timeout_at=start + 5)
        assert not timed_out
        # settle=0 must not wait for any window; anything near the 200 ms quiet period would show it.
        assert now() - start < 0.15
        assert link.take() == "x"


async def test_settle_zero_with_empty_buffer_returns_empty() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        start = now()
        await link.collect(settle=0, timeout_at=start + 5)
        assert now() - start < 0.15
        assert link.take() == ""


# --- Cancellation and the event loop (R11, R12, K3) ------------------------------------------


async def test_cancelled_collect_keeps_bytes() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.play([(0.0, b"keep")])
        task = asyncio.ensure_future(link.collect(settle=None, timeout_at=now() + 5))
        await wait_until(lambda: link.has_retained())
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert link.take() == "keep"


async def test_collect_does_not_block_the_event_loop() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        task = asyncio.ensure_future(ticker())
        try:
            # A 0.6 s window with a 50 ms ticker should tick about 12 times; 5 proves the loop ran.
            await link.collect(settle=0.6, timeout_at=now() + 5)
        finally:
            task.cancel()
        assert ticks >= 5


# --- Cleanup across reads (CL4, CL5, CL8, N5) ------------------------------------------------
# These tests wait for the bytes themselves, then call take(). They do not depend on a window.


async def test_escape_split_across_reads_is_stripped() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"\x1b[3")
        await wait_until(lambda: bytes(link.raw) == b"\x1b[3")
        svc.write(b"1mHi")
        await wait_until(lambda: bytes(link.raw) == b"\x1b[31mHi")
        assert link.take() == "Hi"


async def test_held_escape_is_returned_by_the_next_call() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"abc\x1b[")
        await wait_until(lambda: bytes(link.raw) == b"abc\x1b[")
        assert link.take() == "abc"
        assert bytes(link.raw) == b"\x1b["

        svc.write(b"31mdef")
        await wait_until(lambda: bytes(link.raw) == b"\x1b[31mdef")
        assert link.take() == "def"


async def test_utf8_split_across_reads_is_intact() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"caf\xc3")
        await wait_until(lambda: bytes(link.raw) == b"caf\xc3")
        assert link.take() == "caf"
        assert bytes(link.raw) == b"\xc3"

        svc.write(b"\xa9!")
        await wait_until(lambda: bytes(link.raw) == b"\xc3\xa9!")
        assert link.take() == "é!"


async def test_invalid_bytes_are_replaced_not_raised() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"\xff ok")
        await wait_until(lambda: bytes(link.raw) == b"\xff ok")
        assert link.take() == "� ok"


async def test_telnet_iac_bytes_are_not_stripped() -> None:
    async with FakeService() as svc, open_link(svc) as (link, _):
        svc.write(b"\xff\xfb\x01")
        await wait_until(lambda: bytes(link.raw) == b"\xff\xfb\x01")
        assert link.take() == "��\x01"
