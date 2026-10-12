"""ANSI/VT escape stripping.

Operates on bytes: ESC (0x1B) and every byte that makes up an escape sequence are ASCII, so the
scan is safe before UTF-8 decoding.
"""

ESC = 0x1B
BEL = 0x07
BACKSLASH = 0x5C

# An unfinished sequence longer than this is dropped rather than held. Keeps a malformed,
# unterminated OSC or CSI from holding the stream forever.
MAX_PENDING = 64

# ESC followed by one of these opens a string sequence (OSC, DCS, SOS, PM, APC), ended by ST.
_STRING_INTRODUCERS = frozenset(b"]PX^_")


def strip(data: bytes) -> tuple[bytes, bytes]:
    """Remove every ESC-introduced sequence.

    Returns (clean, pending). pending is an unfinished sequence at the very end of data, held
    back for the next read. It is empty when data ends cleanly, and when the unfinished sequence
    is longer than MAX_PENDING (then it is dropped).
    """
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if data[i] != ESC:
            out.append(data[i])
            i += 1
            continue
        end = _sequence_end(data, i)
        if end is None:
            pending = data[i:]
            if len(pending) > MAX_PENDING:
                return bytes(out), b""
            return bytes(out), bytes(pending)
        i = end
    return bytes(out), b""


def _sequence_end(data: bytes, start: int) -> int | None:
    """Index just past the sequence that starts with the ESC at start.

    Returns None when the data ends before the sequence does. When a sequence is cut short by a
    byte that cannot belong to it, returns that byte's index, so the byte is kept.
    """
    n = len(data)
    j = start + 1
    if j >= n:
        return None
    introducer = data[j]

    if introducer == ord("["):
        # CSI: parameters and intermediates, then one final byte in 0x40-0x7E.
        j += 1
        while j < n and 0x20 <= data[j] <= 0x3F:
            j += 1
        if j >= n:
            return None
        if 0x40 <= data[j] <= 0x7E:
            return j + 1
        return j

    if introducer in _STRING_INTRODUCERS:
        # String sequence: ends at BEL or at ST (ESC \). An ESC that is not ST starts a new sequence.
        j += 1
        while j < n:
            if data[j] == BEL:
                return j + 1
            if data[j] == ESC:
                if j + 1 >= n:
                    return None
                if data[j + 1] == BACKSLASH:
                    return j + 2
                return j
            j += 1
        return None

    if 0x20 <= introducer <= 0x2F:
        # Intermediates, then one final byte: ESC ( B, ESC # 8, and similar.
        j += 1
        while j < n and 0x20 <= data[j] <= 0x2F:
            j += 1
        if j >= n:
            return None
        if 0x30 <= data[j] <= 0x7E:
            return j + 1
        return j

    if introducer == ESC:
        # Drop the first ESC; the second is handled by the caller's loop.
        return j

    if 0x30 <= introducer <= 0x7E:
        # Two-byte sequence: ESC followed by one final byte (ESC =, ESC Z, ESC 7).
        return j + 1

    # ESC followed by a control or high byte: drop only the ESC.
    return j
