#!/usr/bin/env bash
# Start a temporary llama.cpp server for lessons 4-6, bound to localhost only.
#
#   scripts/start_server.sh MODEL.gguf [PORT]
#
# LLAMA_SERVER can point at the llama-server binary if it is not on PATH. The server stops by
# itself after LESSON_SERVER_SECONDS (default 2400), so a forgotten one cannot hold GPU memory.
# LESSON_PARALLEL (default 1) sets how many requests it serves at once, and LESSON_CTX (default
# 4096) the total context shared between them; the capstone's batching script needs 16 and 16384.
# LESSON_GPU_LAYERS (default 99, meaning all of them) sets how many layers go on the GPU: 0 runs
# the whole model on the CPU, and a number in between splits it when the GPU's memory is too
# small for all 32. A llama.cpp built without GPU support ignores it.
# Stop it sooner with:  kill "$(cat /tmp/llm-bytes-up-server.pid)"
set -euo pipefail

model="${1:?usage: start_server.sh MODEL.gguf [PORT]}"
port="${2:-8099}"
bin="${LLAMA_SERVER:-llama-server}"
seconds="${LESSON_SERVER_SECONDS:-2400}"
pidfile=/tmp/llm-bytes-up-server.pid
log=/tmp/llm-bytes-up-server.log

gpu_layers="${LESSON_GPU_LAYERS:-99}"

if [[ ! -f "$model" ]]; then
    echo "no such model file: $model" >&2
    exit 1
fi
if [[ ! "$gpu_layers" =~ ^[0-9]+$ ]]; then
    echo "LESSON_GPU_LAYERS must be a whole number of layers, not '$gpu_layers'" >&2
    exit 1
fi
# A llama.cpp build that keeps its shared libraries beside the binary needs them on the path.
if [[ "$bin" == */* ]]; then
    LD_LIBRARY_PATH="$(dirname "$bin")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi

# Build the launcher from whatever this system has. On Linux that is setsid nohup timeout, as
# before. setsid (util-linux) only detaches from the terminal's session, so it is skipped where
# it does not exist (macOS, Git Bash); nohup still guards against the terminal closing.
# timeout is GNU coreutils: Homebrew's coreutils installs it as gtimeout, and without either a
# one-line perl sets an alarm and execs the server, which the alarm then kills on time.
# nohup is always in the array, so it is never empty (bash 3.2 rejects an empty "${a[@]}" under set -u).
run=()
command -v setsid >/dev/null 2>&1 && run+=(setsid)
run+=(nohup)
if command -v timeout >/dev/null 2>&1; then
    run+=(timeout "$seconds")
elif command -v gtimeout >/dev/null 2>&1; then
    run+=(gtimeout "$seconds")
else
    run+=(perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$seconds")
fi

"${run[@]}" "$bin" -m "$model" --host 127.0.0.1 --port "$port" \
    -c "${LESSON_CTX:-4096}" -ngl "$gpu_layers" --parallel "${LESSON_PARALLEL:-1}" >"$log" 2>&1 </dev/null &
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
