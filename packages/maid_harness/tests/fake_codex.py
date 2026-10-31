"""A stand-in for `codex exec` in tests: same argv shape and JSONL events, and like Codex it starts the maid MCP
server from `-c mcp_servers.maid.command/args` and calls its tools over stdio.

The scenario is a word in the prompt: SCENARIO:done | slow | ask | perm | exit | sleep | crash.
On `exec resume <thread> <message>` it submits a result that echoes the message and the filesystem grants.
It also writes env.json (its environment) and argv.json into the workspace, for the tests to inspect.
"""

import json
import os
import subprocess
import sys
import time


class MCP:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.n = 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18"})

    def rpc(self, method, params):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def call(self, name, args):
        return self.rpc("tools/call", {"name": name, "arguments": args})


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    argv = sys.argv[1:]
    cfg, raw = {}, {}
    for i, a in enumerate(argv):
        if a == "-c":
            k, v = argv[i + 1].split("=", 1)
            raw[k] = v
            try:
                cfg[k] = json.loads(v)
            except json.JSONDecodeError:
                cfg[k] = v
    json.dump(dict(os.environ), open("env.json", "w"))
    json.dump(argv, open("argv.json", "w"))
    mcp = MCP([cfg["mcp_servers.maid.command"], *cfg["mcp_servers.maid.args"]])
    resume = argv[:2] == ["exec", "resume"]
    message = argv[-1]
    emit({"type": "thread.started", "thread_id": argv[-2] if resume else "thread-123"})
    emit({"type": "turn.started"})
    if resume:
        mcp.call("submit_result", {"user_summary": f"resumed with: {message}",
                                   "details": raw.get("permissions.maid.filesystem", "")})
    elif "SCENARIO:done" in message:
        emit({"type": "item.completed", "item": {"type": "command_execution", "command": "echo hi"}})
        mcp.call("report_progress", {"percent": 50, "text": "halfway"})
        mcp.call("submit_result", {"user_summary": "It is 19 degrees in Tokyo.", "details": "full list: A 1980 yen, B 2480 yen"})
    elif "SCENARIO:ask" in message:
        mcp.call("ask_user", {"question": "Which city?"})
    elif "SCENARIO:perm" in message:
        mcp.call("request_permission", {"kind": "read_path", "target": "/tmp/elsewhere",
                                        "what": "read /tmp/elsewhere", "why": "you asked for it"})
    elif "SCENARIO:slow" in message:  # long enough to kill the Maid in between
        time.sleep(1.5)
        mcp.call("submit_result", {"user_summary": "done while you were away"})
    elif "SCENARIO:sleep" in message:
        time.sleep(30)
    elif "SCENARIO:crash" in message:
        print("boom", file=sys.stderr)
        sys.exit(3)
    # SCENARIO:exit: just stop, without submit_result
    emit({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2}})


if __name__ == "__main__":
    main()
