#!/usr/bin/env bash
# Regenerate the cargo-fuzz corpora from the checked-in fixture corpora
# (audit P0-75/P0-81): seeds come from `compat/kilo-v756/` and
# `fixtures/providers/`, plus canonical text seeds for targets that have no
# fixture file of their own (ACP frames, tool-call text, path/rule/model
# strings).
#
# Env:
#   FUZZ_CORPUS_DIR   corpus dir override (default: fuzz/corpus)
#
# Idempotent and offline. Exit non-zero only when a seed cannot be written.

set -u
set -o pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="${FUZZ_CORPUS_DIR:-$ROOT/fuzz/corpus}"
SEEDED=0

seed_copy() {
    target="$1"
    shift
    dest="$CORPUS/$target"
    mkdir -p "$dest" || return 1
    for src in "$@"; do
        [ -f "$src" ] || continue
        cp "$src" "$dest/$(basename "$src")" || return 1
        SEEDED=$((SEEDED + 1))
    done
}

seed_text() {
    target="$1"
    name="$2"
    text="$3"
    dest="$CORPUS/$target"
    mkdir -p "$dest" || return 1
    printf '%s' "$text" >"$dest/$name" || return 1
    SEEDED=$((SEEDED + 1))
}

# --- compat + provider fixtures feed the DTO/event/SSE/framing targets ----
seed_copy compat_dto "$ROOT"/compat/kilo-v756/*.json
seed_copy event_payload "$ROOT"/compat/kilo-v756/*.json
seed_copy sse_frame "$ROOT"/compat/kilo-v756/*.json
seed_copy line_framing "$ROOT"/compat/kilo-v756/*.json
seed_copy line_framing "$ROOT"/fixtures/providers/*.json
seed_copy sse_frame "$ROOT"/fixtures/providers/*.json
seed_copy tokenizer_pricing "$ROOT"/fixtures/providers/ollama-api-show-qwen3.8.json

# --- ACP frames: Content-Length framed JSON-RPC bodies ---------------------
acp_init='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}'
seed_text acp_frame initialize.frame \
    "$(printf 'Content-Length: %s\r\n\r\n%s' "${#acp_init}" "$acp_init")"
acp_update='{"jsonrpc":"2.0","id":null,"method":"session/update","params":{"sessionId":"sess-1","update":{"sessionUpdate":"agent_message_chunk"}}}'
seed_text acp_frame update.frame \
    "$(printf 'Content-Length: %s\r\n\r\n%s' "${#acp_update}" "$acp_update")"
seed_text acp_frame hostile-header.frame 'Content-Length: 999999999\r\n\r\n'

# --- tool-call JSON repair/parse -------------------------------------------
seed_text tool_json perfect.json '{"name":"read_file","input":{"path":"src/lib.rs"}}'
seed_text tool_json fenced.md '```json
{"name":"write_file","input":{"path":"a.rs","content":"x"},}
```'
seed_text tool_json single-quoted.txt "{'name': 'exec', 'arguments': {'cmd': 'echo hi'}}"

# --- path normalization -----------------------------------------------------
seed_text path_normalization relative.txt 'sub/deep.txt'
seed_text path_normalization traversal.txt '../../etc/passwd'
seed_text path_normalization absolute.txt '/etc/passwd'

# --- destination policy -----------------------------------------------------
seed_text destination_policy rules.txt 'api.example.com
*.internal.example.com:8443
https://host.example.com:443'
seed_text destination_policy hostile.txt 'not a host:999999
http:///missing-host
::1'

# --- tokenizer/pricing ------------------------------------------------------
seed_text tokenizer_pricing model.txt 'openai/gpt-4o-2024-08-06'
seed_text tokenizer_pricing snapshot.json \
    '{"quote":{"input":10000000,"output":30000000,"cache_read":0,"cache_write":0},"authority":"exact","epoch":1,"source_id":"fuzz-seed"}'

printf '[fuzz-seeds] seeded %s corpus files into %s\n' "$SEEDED" "$CORPUS"
