import pytest

from text_adventure_mcp import ansi


@pytest.mark.parametrize(
    ("data", "clean"),
    [
        # CSI: colour, cursor movement, erase
        (b"\x1b[31mred\x1b[0m", b"red"),
        (b"\x1b[2;5Hhere", b"here"),
        (b"\x1b[2Kline", b"line"),
        (b"\x1b[?25lhidden", b"hidden"),
        # OSC: window title, ended by BEL or by ST
        (b"\x1b]0;title\x07x", b"x"),
        (b"\x1b]0;title\x1b\\x", b"x"),
        # String sequences other than OSC (DCS, SOS, PM, APC)
        (b"\x1bPdata\x1b\\x", b"x"),
        (b"\x1b_app\x07x", b"x"),
        # Two- and three-byte sequences
        (b"\x1b=keypad", b"keypad"),
        (b"\x1b(Bplain", b"plain"),
        (b"\x1b#8x", b"x"),
        (b"\x1bZafter", b"after"),
        # ESC ESC: the first ESC goes, the second starts its own sequence
        (b"\x1b\x1b[31mx", b"x"),
        # ESC followed by a control byte: only the ESC goes
        (b"\x1b\nkept", b"\nkept"),
        # CSI cut short by a control byte: the control byte is kept
        (b"\x1b[\nkept", b"\nkept"),
        # An OSC interrupted by a new ESC: the OSC ends, the CSI is stripped
        (b"\x1b]0;t\x1b[31mX", b"X"),
        # No escapes at all
        (b"plain text\r\n", b"plain text\r\n"),
    ],
)
def test_strip_removes_sequences(data: bytes, clean: bytes) -> None:
    assert ansi.strip(data) == (clean, b"")


def test_non_ascii_bytes_pass_through() -> None:
    assert ansi.strip("é\x1b[1m".encode()) == ("é".encode(), b"")


@pytest.mark.parametrize(
    ("data", "clean", "pending"),
    [
        (b"ab\x1b", b"ab", b"\x1b"),
        (b"ab\x1b[31", b"ab", b"\x1b[31"),
        (b"ab\x1b(", b"ab", b"\x1b("),
        (b"\x1b]0;tit", b"", b"\x1b]0;tit"),
        (b"\x1b]0;t\x1b", b"", b"\x1b]0;t\x1b"),
    ],
)
def test_unfinished_sequence_is_held(data: bytes, clean: bytes, pending: bytes) -> None:
    assert ansi.strip(data) == (clean, pending)


def test_pending_at_limit_is_held() -> None:
    data = b"\x1b]" + b"x" * (ansi.MAX_PENDING - 2)
    assert len(data) == ansi.MAX_PENDING
    assert ansi.strip(data) == (b"", data)


def test_pending_over_limit_is_dropped() -> None:
    data = b"\x1b]" + b"x" * (ansi.MAX_PENDING - 1)
    assert len(data) == ansi.MAX_PENDING + 1
    assert ansi.strip(b"ok" + data) == (b"ok", b"")


def test_held_sequence_completes_on_next_read() -> None:
    clean, pending = ansi.strip(b"abc\x1b[")
    assert (clean, pending) == (b"abc", b"\x1b[")
    assert ansi.strip(pending + b"31mdef") == (b"def", b"")
