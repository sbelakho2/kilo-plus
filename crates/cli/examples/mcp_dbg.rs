use std::sync::Arc;
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("m.py");
    // Use the EXACT bridge-test mock
    std::fs::write(
        &script,
        r#"import sys, json
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
send({"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock","version":"1"}}})
while True:
    msg = read()
    if msg is None:
        break
    mid = msg.get("id")
    m = msg.get("method")
    if m == "tools/list":
        send({"jsonrpc":"2.0","id":mid,"result":{"tools":[{"name":"mock_echo","description":"echo text back","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}})
    elif m == "tools/call":
        args = msg.get("params",{}).get("arguments",{})
        text = args.get("text","")
        if text == "boom":
            send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"exploded"}],"isError":True}})
        else:
            send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"echo:"+text}],"isError":False}})
    else:
        send({"jsonrpc":"2.0","id":mid,"result":{}})
"#,
    )
    .unwrap();
    let cas = Arc::new(kilop_cas::Cas::open(dir.path().join("cas")).unwrap());
    let sup = kilop_terminal::ProcessSupervisor::new(cas);
    let cfg = kilop_mcp::McpConfig {
        name: "mock".into(),
        command: "python3".into(),
        args: vec![script.to_str().unwrap().into()],
        env: vec![],
    };
    let r = kilop_mcp::McpServer::connect(cfg, sup).await;
    match r {
        Ok(s) => {
            println!("connect ok");
            let tools = s.list_tools().await.unwrap();
            println!(
                "tools: {:?}",
                tools.iter().map(|t| t.name.clone()).collect::<Vec<_>>()
            );
            let r = s
                .call_tool(
                    "mock_echo",
                    serde_json::json!({"text": "hi"}),
                    std::time::Duration::from_secs(5),
                )
                .await;
            println!("call: {:?}", r.map(|x| x.content));
        }
        Err(e) => println!("connect err: {e}"),
    }
}
