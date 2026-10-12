import text_adventure_mcp
from text_adventure_mcp import ansi, server, session, wire


def test_package_imports() -> None:
    assert text_adventure_mcp.__version__ == "0.1.0"
    assert ansi.MAX_PENDING == 64
    assert callable(server.main)
    assert session is not None
    assert wire is not None
