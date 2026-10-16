import re

import text_adventure_mcp
from text_adventure_mcp import ansi, server, session, wire


def test_package_imports() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", text_adventure_mcp.__version__)
    assert ansi.MAX_PENDING == 64
    assert callable(server.main)
    assert session is not None
    assert wire is not None
