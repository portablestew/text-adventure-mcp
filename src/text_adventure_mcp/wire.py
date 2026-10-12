"""Command normalization, UTF-8 decoding, receive windowing, and the per-connection link.

The link owns its socket, a reader task that drains it, and the buffer of received bytes. Bytes
stay in the buffer until a call hands them out with take(), so data that arrives between calls,
or during a call that is cancelled or times out, is never lost.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable

from text_adventure_mcp import ansi

# Turn-mode completion: quiet for QUIET_S, or CAP_S since the first byte, whichever is first.
QUIET_S = 0.2
CAP_S = 1.0

READ_SIZE = 65536

# A line break not already preceded by CR.
_BARE_LF = re.compile(rb"(?<!\r)\n")


def normalize_command(command: str) -> bytes:
    """Line-normalize a command for sending. Always ends in CRLF, so the service always gets a turn."""
    data = _BARE_LF.sub(b"\r\n", command.encode("utf-8", errors="replace"))
    if not data.endswith(b"\r\n"):
        data += b"\r\n"
    return data


def decode(data: bytes) -> tuple[str, bytes]:
    """Decode UTF-8, replacing invalid bytes. Returns (text, incomplete_tail).

    incomplete_tail is a multi-byte sequence cut off at the end of data. It is returned as bytes
    so the caller can keep it until the rest arrives.
    """
    tail = _incomplete_utf8_tail(data)
    body = data[: len(data) - tail] if tail else data
    return body.decode("utf-8", errors="replace"), data[len(body) :]


def _incomplete_utf8_tail(data: bytes) -> int:
    """Number of trailing bytes that begin a multi-byte character not yet complete."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 == 0x80:
            # Continuation byte: keep looking for its lead byte.
            continue
        if byte & 0xE0 == 0xC0:
            need = 2
        elif byte & 0xF0 == 0xE0:
            need = 3
        elif byte & 0xF8 == 0xF0:
            need = 4
        else:
            return 0
        return back if need > back else 0
    return 0


class LinkLost(Exception):
    """The socket failed or closed while a call was using it."""


class Link:
    """One TCP connection: the socket, its reader task, and the buffer of unhanded bytes."""

    raw: bytearray
    closed: bool

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        on_close: Callable[[Link], None],
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._on_close = on_close
        self.raw = bytearray()
        self.closed = False
        # Bumped on every chunk, so collect() can tell arrivals apart without relying on the event.
        self._arrivals = 0
        self._changed = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._read_loop())

    @classmethod
    async def open(
        cls,
        host: str,
        port: int,
        on_close: Callable[[Link], None],
    ) -> Link:
        reader, writer = await asyncio.open_connection(host, port)
        return cls(reader, writer, on_close)

    async def send(self, payload: bytes) -> None:
        """Write one payload. Raises LinkLost if the socket is gone."""
        if self.closed:
            raise LinkLost("connection is closed")
        try:
            self._writer.write(payload)
            await self._writer.drain()
        except OSError as exc:
            self._mark_closed()
            raise LinkLost(str(exc)) from exc

    async def collect(self, *, settle: float | None, timeout_at: float) -> bool:
        """Wait for the receive window. Returns True if the call timed out before completing.

        Does not return bytes; the caller calls take() afterwards. settle=None selects turn mode
        (complete on quiet or cap). A number selects stream mode (fixed window of that many
        seconds). Both end no later than timeout_at. Raises LinkLost if the socket closes first.
        """
        loop = asyncio.get_running_loop()
        if settle is None:
            return await self._collect_turn(loop, timeout_at)
        return await self._collect_stream(loop, settle, timeout_at)

    def take(self) -> str:
        """Hand out everything received so far, as text, and clear the buffer.

        An unfinished escape sequence or multi-byte character at the end stays in the buffer for
        the next call.
        """
        clean, pending = ansi.strip(bytes(self.raw))
        text, utf8_tail = decode(clean)
        self.raw = bytearray(utf8_tail + pending)
        return text

    def has_retained(self) -> bool:
        return bool(self.raw)

    async def aclose(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._mark_closed()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()

    async def _collect_turn(self, loop: asyncio.AbstractEventLoop, timeout_at: float) -> bool:
        start = loop.time()
        first: float | None = None
        last: float | None = None
        seen = self._arrivals
        if self.raw:
            # A backlog counts as the first byte already seen (the window still runs its quiet period).
            first = last = start
        while True:
            self._raise_if_closed()
            if self._arrivals != seen:
                seen = self._arrivals
                last = loop.time()
                if first is None:
                    first = last
            now = loop.time()
            done_at: float | None = None
            if first is not None and last is not None:
                done_at = min(last + QUIET_S, first + CAP_S)
                if now >= done_at:
                    return False
            if now >= timeout_at:
                return True
            wake = timeout_at if done_at is None else min(done_at, timeout_at)
            await self._wait_changed(wake - now)

    async def _collect_stream(
        self,
        loop: asyncio.AbstractEventLoop,
        settle: float,
        timeout_at: float,
    ) -> bool:
        window_end = loop.time() + settle
        end = min(window_end, timeout_at)
        while True:
            self._raise_if_closed()
            now = loop.time()
            if now >= end:
                return end < window_end
            await self._wait_changed(end - now)

    async def _wait_changed(self, seconds: float) -> None:
        """Sleep until the next arrival, close, or the given number of seconds."""
        self._changed.clear()
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self._changed.wait()

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise LinkLost("connection closed during call")

    async def _read_loop(self) -> None:
        try:
            while True:
                chunk = await self._reader.read(READ_SIZE)
                if not chunk:
                    break
                self.raw.extend(chunk)
                self._arrivals += 1
                self._changed.set()
        except OSError:
            pass
        finally:
            self._mark_closed()

    def _mark_closed(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._changed.set()
        self._writer.close()
        self._on_close(self)
