"""A stand-in for `codex exec` in tests: same argv shape, same JSONL events, calls the real maid MCP URL.

The scenario is a word in the prompt: SCENARIO:done | ask | perm | exit | sleep | crash.
On `exec resume <thread> <message>` it submits a result that echoes the message (and any writable roots).
"""

import json
import sys
import time
import urllib.request


def mcp(url, name, args):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    argv = sys.argv[1:]
    cfg = {}
    for i, a in enumerate(argv):
        if a == "-c":
            k, v = argv[i + 1].split("=", 1)
            cfg[k] = json.loads(v)
    url = cfg["mcp_servers.maid.url"]
    resume = argv[:2] == ["exec", "resume"]
    message = argv[-1]
    emit({"type": "thread.started", "thread_id": argv[-2] if resume else "thread-123"})
    emit({"type": "turn.started"})
    if resume:
        roots = cfg.get("sandbox_workspace_write.writable_roots", [])
        mcp(url, "submit_result", {"user_summary": f"resumed with: {message}", "details": json.dumps(roots)})
    elif "SCENARIO:done" in message:
        emit({"type": "item.completed", "item": {"type": "command_execution", "command": "echo hi"}})
        mcp(url, "report_progress", {"percent": 50, "text": "halfway"})
        mcp(url, "submit_result", {"user_summary": "It is 19 degrees in Tokyo."})
    elif "SCENARIO:ask" in message:
        mcp(url, "ask_user", {"question": "Which city?"})
    elif "SCENARIO:perm" in message:
        mcp(url, "request_permission", {"kind": "write_path", "target": "/tmp/elsewhere",
                                        "what": "write /tmp/elsewhere/x.txt", "why": "you asked for it there"})
    elif "SCENARIO:sleep" in message:
        time.sleep(30)
    elif "SCENARIO:crash" in message:
        print("boom", file=sys.stderr)
        sys.exit(3)
    # SCENARIO:exit: just stop, without submit_result
    emit({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}})


if __name__ == "__main__":
    main()
