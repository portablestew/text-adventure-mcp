# Text Adventure MCP — Requirements & Goals

## Goal

A minimal MCP server that lets an agent converse with a line-oriented text service over TCP: a
Zork-style game, a telnet MUD, or any automation that exposes a text command interface.

The server is a dumb pipe. It holds the socket, terminates lines, strips escape codes, and hands
back whatever the service said. All interpretation of the text stays with the agent.

## Why MCP

The connection is stateful — the service remembers who you are and where you are between commands.
The server exists to keep that socket alive across tool calls. That is the whole reason this is an
MCP server rather than a script.

## Tool surface

Exactly one tool. No lifecycle tools.

### `text_adventure`

| param        | type             | required | default          | meaning                                                     |
|--------------|------------------|----------|------------------|-------------------------------------------------------------|
| `command`    | string           | yes      | —                | Text to send. Line-normalized before sending; empty sends a blank line. |
| `address`    | string           | no       | `127.0.0.1:3000` | `host:port` of the service. Identifies the connection.      |
| `settle_ms`  | integer, optional | no       | not set          | Fixed collection window, in milliseconds. Not set = wait for a complete response. |
| `timeout_ms` | integer, optional | no       | `20000`, or `20000 + settle_ms` | Hard cap on the call, in milliseconds.        |

`settle_ms` and `timeout_ms` are optional whole-millisecond integers from `0` to `300000`, not
nullable fields — the caller omits them. An explicit `null` is accepted and means the same as
omitted, since typed proxies routinely emit `null` for an unset field. In Python they are
`int | None = None`.

## Behavior

### Connection

- One TCP connection per `address`, held open for the life of the server.
- The registry is keyed by the `address` string exactly as given. Two spellings of the same
  service (`localhost:3000`, `127.0.0.1:3000`) are two connections.
- The first call with a given `address` connects; later calls reuse the socket.
- If a socket errors or closes, drop it from the registry. The next call to that `address`
  reconnects. No explicit close or cleanup.
- Calls against the same `address` are serialized; they never interleave reads on one socket.

### Sending

Every call is a turn: the server always sends at least a bare `\r\n`, so the service always has
something to reply to. Each `command` is sent as one payload after line normalization:

1. Convert any `\n` not already preceded by `\r` to `\r\n`.
2. If the result does not end with `\r\n`, append `\r\n`.

An empty `command` therefore goes out as `"\r\n"` — a blank line, the equivalent of pressing
Enter. There is no send-nothing mode.

### Receiving

The return value is always **everything received on that connection since the previous call on that
connection**, including anything that arrived while the agent was doing other work. `settle_ms` and
`timeout_ms` only decide how long to wait before returning — never what is returned.

The wait runs on every call, including when output is already buffered. A call with a backlog still
waits the full window, then returns the backlog together with whatever arrived meanwhile.

- `settle_ms` not set (turn-based, e.g. Zork): wait for the first byte, bounded by `timeout_ms`, then
  collect until the response is complete. A response is complete when nothing has arrived for
  200 ms, or when collection has run for 1000 ms since the first byte — whichever comes first.
  Both figures are internal constants, not caller-configurable. The 1000 ms cap exists so a
  service that never goes quiet still returns.
- `settle_ms: <milliseconds>` (streaming, e.g. an online MUD): collect for exactly that many
  milliseconds, then return. This is a fixed window, not a debounce. `settle_ms: 0` returns
  immediately with whatever is already buffered. The 1000 ms cap does not apply; the caller chose
  the window.
- `timeout_ms`: hard cap on the entire call, including connection setup. On expiry, return whatever
  has been received so far — an empty string if nothing has.

A new connection's buffer starts at connect time, so the service's banner and login prompt come
back with the first call.

### Status text

A result is status lines followed by service output. Status lines cover only what the agent cannot
infer from the service text itself, one per line, prefixed to keep them distinct from game output:

```
[status] connected to 127.0.0.1:3000
Welcome to Zork I...
```

The reported set is closed:

- `[status] connected to <address>` — this call opened a connection.
- `[status] connection to <address> lost` — the socket failed or closed mid-call.
- `[status] timed out after <n>s` — `timeout_ms` expired before the service finished.

Nothing else is reported. Routine sends, empty responses, and normal completions are silent. Status
lines travel with the result whether it is returned normally or as a tool error.

### Cleanup of the text

- ANSI/VT escape sequences are stripped with one rule: anything introduced by `ESC` is removed.
  That covers CSI (colour, cursor movement, erases), OSC (window titles and the like), and
  two-character sequences alike.
- Bytes are decoded as UTF-8; undecodable bytes are replaced, not raised.

### Errors

- A socket that errors or closes is dropped from the registry; the next call to that `address`
  reconnects.
- Failures — connect, send, or mid-call socket loss — surface as MCP tool errors. The error message
  carries whatever output had been received and not yet returned, so no text is ever thrown away.
- Invalid parameter values (`settle_ms` or `timeout_ms` outside `0..300000`) are tool errors.

## Constraints

- Transport: stdio.
- Python 3.11 or newer.
- Implementation is async end to end (`asyncio`). A blocking socket would stall the server.
- Logs go to stderr only. On stdio transport, stdout is the protocol channel.
- No game-specific or protocol-specific logic in the server.

## Non-goals

- No connection close, reset, or cleanup tool. No idle expiry.
- No output truncation, pagination, or draining. Return everything.
- No structured output. Results are plain text: status lines plus service output.
- No prompt detection or response-completion heuristics beyond the 200 ms quiet period and its
  1000 ms cap.
- No telnet negotiation (IAC DO/WILL/WONT/DONT) and no IAC stripping. Raw TCP is enough.
- No TLS.
- No splitting multi-line commands into separate sends.
- No authentication or configuration UI.

## Design stage scope

Everything above is settled. Deliberately unspecified here, to be decided during design:

- Tool description text and the JSON schema the model sees.
- Internal module boundaries, async structure, and the shape of the connection registry.
- Exact error message wording.
- Test plan beyond the two service styles named under Package shape.

## Package shape

```
text-adventure-mcp/
├── pyproject.toml            # [project.scripts] text-adventure-mcp
├── REQUIREMENTS.md
├── src/text_adventure_mcp/
│   ├── __init__.py
│   ├── server.py             # FastMCP app and the text_adventure tool (thin)
│   ├── session.py            # connection registry, per-address lock
│   ├── wire.py               # send, receive, settle/timeout windowing
│   └── ansi.py               # escape stripping
└── tests/
    ├── fakes.py              # turn-based and streaming fake services
    ├── test_wire.py
    └── test_server.py
```

- SDK: official `mcp` package, `FastMCP`.
- Tests cover both service styles: one that replies and waits for the next move, one that streams.

## Safety note

This is an arbitrary outbound TCP client and can reach anything the host can reach, including
internal services. That is the point. The README should say so.
