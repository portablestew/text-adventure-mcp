# text-adventure-mcp

A minimal MCP server that lets an agent converse with a line-oriented text service over TCP: a
Zork-style game, a telnet MUD, or any automation that exposes a text command interface.

The server is a dumb pipe. It holds the socket, terminates lines, strips escape codes, and hands
back whatever the service said. All interpretation of the text stays with the agent.

The connection is stateful, so the server keeps one socket open per `address` across tool calls.

## Tool

One tool, `text_adventure`:

| parameter | type             | required | default          | meaning |
|-----------|------------------|----------|------------------|---------|
| `command` | string           | yes      | —                | Text to send. Empty sends a blank line. |
| `address` | string           | no       | `127.0.0.1:3000` | `host:port` of the service. Identifies the connection. |
| `settle`  | number, optional | no       | not set          | Fixed collection window, in seconds. Not set = wait for a complete response. |
| `timeout` | number, optional | no       | `20`, or `20 + settle` | Hard cap on the call, in seconds. |

## Safety

This is an arbitrary outbound TCP client and can reach anything the host can reach, including
internal services. That is the point. Do not expose it to untrusted callers, and do not point it at
services you are not authorized to use.

## Install and run

Requires Python 3.11 or newer.

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
.venv/Scripts/text-adventure-mcp.exe      # stdio transport; blocks waiting for a client
```

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

## Naming note

The design notes call the SDK class `FastMCP`. In the official `mcp` 2.x package that class is
named `MCPServer` (`from mcp.server.mcpserver import MCPServer`). This project depends on
`mcp>=2,<3` and uses `MCPServer`. The `mcp.server.fastmcp` module from 1.x is not present in 2.x.
