import sys, json

def send(obj):
    body = json.dumps(obj).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()

def read():
    cl = 0
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if line.lower().startswith(b"content-length:"):
            cl = int(line.split(b":")[1])
    if cl == 0:
        return {}
    return json.loads(sys.stdin.buffer.read(cl))

send({"jsonrpc": "2.0", "id": 0, "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "serverInfo": {"name": "mock", "version": "1"},
}})
while True:
    msg = read()
    if msg is None:
        break
    mid = msg.get("id")
    m = msg.get("method")
    if m == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
            "name": "mock_echo",
            "description": "echo text back",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }]}})
    elif m == "tools/call":
        args = msg.get("params", {}).get("arguments", {})
        text = args.get("text", "")
        if text == "boom":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "exploded"}], "isError": True,
            }})
        else:
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "echo:" + text}], "isError": False,
            }})
    else:
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
