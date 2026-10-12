"""Fake line-oriented services for tests. Real TCP on 127.0.0.1 with an ephemeral port.

Every fake records the bytes it receives and the number of connections it accepted, so tests can
check what went over the wire, not just what came back.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from types import TracebackType
from typing import Self

# A schedule is a list of (delay_seconds_before_write, bytes_to_write) pairs.
Schedule = list[tuple[float, bytes]]


class FakeService:
    """Accepts connections and records what they send. Subclasses react in on_data()."""

    def __init__(self, banner: bytes = b"") -> None:
        self.banner = banner
        self.received = bytearray()
        self.accepts = 0
        self.port = 0
        self._server: asyncio.Server | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._play_task: asyncio.Task[None] | None = None

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}"

    async def __aenter__(self) -> Self:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._play_task is not None:
            self._play_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, OSError):
                await self._play_task
        if self._writer is not None:
            self._writer.close()
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def on_data(self, data: bytes) -> None:
        """Called for each chunk the client sends. Override to reply."""

    def write(self, data: bytes) -> None:
        """Send bytes to the client on the current connection."""
        assert self._writer is not None, "no client connected"
        self._writer.write(data)

    def close(self) -> None:
        """Graceful close: the client sees EOF."""
        assert self._writer is not None, "no client connected"
        self._writer.close()

    def play(self, schedule: Schedule) -> None:
        """Send scheduled chunks in the background, starting now."""
        self._play_task = asyncio.get_running_loop().create_task(self._play(schedule))

    async def _play(self, schedule: Schedule) -> None:
        for delay, chunk in schedule:
            await asyncio.sleep(delay)
            if self._writer is None or self._writer.is_closing():
                return
            self.write(chunk)
            with contextlib.suppress(OSError):
                await self._writer.drain()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.accepts += 1
        self._writer = writer
        if self.banner:
            writer.write(self.banner)
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                self.received.extend(data)
                await self.on_data(data)
        except OSError:
            pass
        finally:
            writer.close()


class TurnService(FakeService):
    """Replies to each received line after a delay, like a turn-based game such as Zork.

    respond(line) returns the schedule for that reply. Lines are split on CRLF.
    """

    def __init__(
        self,
        respond: Callable[[bytes], Schedule],
        banner: bytes = b"",
    ) -> None:
        super().__init__(banner)
        self._respond = respond
        self._pending = bytearray()

    async def on_data(self, data: bytes) -> None:
        self._pending.extend(data)
        while b"\r\n" in self._pending:
            line, _, rest = bytes(self._pending).partition(b"\r\n")
            self._pending = bytearray(rest)
            await self._run(self._respond(line))

    async def _run(self, schedule: Schedule) -> None:
        for delay, chunk in schedule:
            await asyncio.sleep(delay)
            if self._writer is not None and not self._writer.is_closing():
                self.write(chunk)
                with contextlib.suppress(OSError):
                    await self._writer.drain()


class StreamService(FakeService):
    """Pushes output on its own schedule, like a streaming MUD. Ignores what the client sends."""


async def wait_until(predicate: Callable[[], bool], limit: float = 2.0) -> None:
    """Poll until predicate is true. Waits on the condition, not on a guessed delay."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    while not predicate():
        assert loop.time() < deadline, "condition not reached in time"
        await asyncio.sleep(0.005)
