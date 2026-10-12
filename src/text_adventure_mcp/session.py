"""Connection registry, per-address lock, and the exchange.

One Link per address string, kept open across calls. Calls on one address are serialized so two
reads never interleave on one socket. Every failure path keeps the received text and reports it
with the Failure, so the caller can return it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from text_adventure_mcp.wire import Link, LinkLost

NOT_SENT_REASON = "Not sent: the command was not delivered; send it again."


@dataclass(frozen=True)
class Address:
    host: str
    port: int
    text: str  # exactly as given; the registry key


def parse_address(text: str) -> Address:
    """Split 'host:port'. Raises ValueError for anything that is not a usable address."""
    host, sep, port_text = text.rpartition(":")
    if not sep or not host:
        raise ValueError(f"address must be host:port, for example 127.0.0.1:3000 (got {text!r})")
    if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
        raise ValueError(f"port must be a number from 1 to 65535 (got {text!r})")
    return Address(host=host, port=int(port_text), text=text)


@dataclass
class Outcome:
    status: list[str]  # status lines, in order, without trailing newlines
    text: str  # service output, exactly as decoded


class Failure(Exception):
    """A call that failed. Carries whatever status and text were available."""

    def __init__(self, status: list[str], text: str, reason: str) -> None:
        super().__init__(reason or "; ".join(status))
        self.status = status
        self.text = text
        self.reason = reason


def _connected_line(address: str) -> str:
    return f"[status] connected to {address}"


def _lost_line(address: str) -> str:
    return f"[status] connection to {address} lost"


def _timed_out_line(timeout: float) -> str:
    return f"[status] timed out after {timeout:g}s"


class Session:
    """Holds one Link per address and runs exchanges over them."""

    def __init__(self) -> None:
        self._links: dict[str, Link] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def exchange(
        self,
        address: Address,
        payload: bytes,
        *,
        settle: float | None,
        timeout: float,
    ) -> Outcome:
        """Send payload to address, then collect the reply within the settle or turn window.

        timeout bounds the whole call: lock wait, connect, and collection.
        """
        deadline = asyncio.get_running_loop().time() + timeout

        lock = self._locks.setdefault(address.text, asyncio.Lock())
        try:
            # timeout_at, not wait_for: with timeout=0 an uncontended lock must still be taken.
            async with asyncio.timeout_at(deadline):
                await lock.acquire()
        except TimeoutError:
            return Outcome([_timed_out_line(timeout)], "")
        try:
            return await self._exchange_locked(address, payload, settle, timeout, deadline)
        finally:
            lock.release()

    async def aclose(self) -> None:
        """Close every open link. For shutdown and tests; there is no close tool."""
        links = list(self._links.values())
        self._links.clear()
        for link in links:
            await link.aclose()

    async def _exchange_locked(
        self,
        address: Address,
        payload: bytes,
        settle: float | None,
        timeout: float,
        deadline: float,
    ) -> Outcome:
        key = address.text
        status: list[str] = []

        link = self._links.get(key)
        if link is not None and link.closed:
            # Dropped while idle. Bytes received before the close are handed out with the error,
            # and the command is not sent. Otherwise the stale link is replaced silently below.
            del self._links[key]
            text = link.take()
            if text:
                raise Failure([_lost_line(key)], text, NOT_SENT_REASON)
            link = None

        opened = False
        if link is None:
            try:
                async with asyncio.timeout_at(deadline):
                    link = await Link.open(
                        address.host, address.port, lambda closed: self._on_close(key, closed)
                    )
            except TimeoutError:
                return Outcome([_timed_out_line(timeout)], "")
            except OSError as exc:
                raise Failure([], "", f"could not connect to {key}: {exc}") from exc
            self._links[key] = link
            opened = True

        try:
            try:
                async with asyncio.timeout_at(deadline):
                    await link.send(payload)
            except TimeoutError:
                # The write was still blocked at the deadline, so part of the command may be on the
                # wire and the socket cannot be reused. Drop it and report what it had received.
                text = link.take()
                self._drop(key, link)
                await link.abort()
                if opened:
                    status.append(_connected_line(key))
                status.append(_timed_out_line(timeout))
                return Outcome(status, text)
            timed_out = await link.collect(settle=settle, timeout_at=deadline)
        except LinkLost as exc:
            # The socket is gone. Drop it, and hand out what it had received.
            self._drop(key, link)
            text = link.take()
            lost = [_connected_line(key)] if opened else []
            raise Failure([*lost, _lost_line(key)], text, "") from exc

        text = link.take()
        if opened:
            status.append(_connected_line(key))
        if timed_out:
            status.append(_timed_out_line(timeout))
        return Outcome(status, text)

    def _on_close(self, key: str, link: Link) -> None:
        """Called by the link when its socket closes. Forgets it unless bytes are waiting."""
        if self._links.get(key) is link and not link.has_retained():
            del self._links[key]

    def _drop(self, key: str, link: Link) -> None:
        if self._links.get(key) is link:
            del self._links[key]
