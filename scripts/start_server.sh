#!/usr/bin/env bash
# Start a temporary llama.cpp server for lessons 4-6, bound to localhost only.
#
#   scripts/start_server.sh MODEL.gguf [PORT]
#
# LLAMA_SERVER can point at the llama-server binary if it is not on PATH. The server stops by
# itself after LESSON_SERVER_SECONDS (default 2400), so a forgotten one cannot hold GPU memory.
# LESSON_PARALLEL (default 1) sets how many requests it serves at once, and LESSON_CTX (default
# 4096) the total context shared between them; the capstone's batching script needs 16 and 16384.
# Stop it sooner with:  kill "$(cat /tmp/llm-bytes-up-server.pid)"
set -euo pipefail

model="${1:?usage: start_server.sh MODEL.gguf [PORT]}"
port="${2:-8099}"
bin="${LLAMA_SERVER:-llama-server}"
seconds="${LESSON_SERVER_SECONDS:-2400}"
pidfile=/tmp/llm-bytes-up-server.pid
log=/tmp/llm-bytes-up-server.log

if [[ ! -f "$model" ]]; then
    echo "no such model file: $model" >&2
    exit 1
fi
# A llama.cpp build that keeps its shared libraries beside the binary needs them on the path.
if [[ "$bin" == */* ]]; then
    export LD_LIBRARY_PATH="$(dirname "$bin")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

setsid nohup timeout "$seconds" "$bin" -m "$model" --host 127.0.0.1 --port "$port" \
    -c "${LESSON_CTX:-4096}" -ngl 99 --parallel "${LESSON_PARALLEL:-1}" >"$log" 2>&1 </dev/null &
echo $! >"$pidfile"

for _ in $(seq 1 60); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" | grep -q '"ok"'; then
        echo "llama-server ready on http://127.0.0.1:$port (pid $(cat "$pidfile"), log $log)"
        exit 0
    fi
    sleep 1
done
echo "llama-server did not become ready; see $log" >&2
exit 1
