# Compatibility fixtures

## kilo-v756/

The frozen v7.5.6 wire contract. Golden tests in `kilop-protocol` lock
request/response/SSE/JSON-field-presence/null-behavior/error-code behavior
against these fixtures byte-for-byte. Changing wire behavior requires
updating fixtures here first.

- `startup_line.json` — the frozen stdout line `kilo server listening on http://127.0.0.1:<port>` (replaced the legacy `handshake.json` JSON-handshake payload)
- `hello.json` — public hello response shape
- `create_session.json` — request/response pair
- `messages_page.json` — paged Message/Part shapes with null behavior
- `sse_frames.json` — frozen SSE frames (resume-cursor sequence)
- `global_event.json` — frozen GlobalEvent envelope examples (session_created, message_part_updated, session_next_text_delta)
- `password_auth.json` — `KILO_SERVER_PASSWORD` header semantics (Bearer + x-kilo-server-password; 401 shape; /global/health is public)
- `errors.json` — error-code → HTTP-status mapping
- `provider_list.json` — provider/model capability objects

## jetbrains-712/

Reserved for the frozen JetBrains 7.1.2 split-mode fixture corpus (shared/
frontend/backend).
