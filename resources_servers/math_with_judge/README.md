# Math With Judge Environment

Evaluates model responses on math problems using the Hugging Face **Math-Verify**
library and an LLM judge whose hosting is **not managed by NeMo-Gym**. The YAML
config supplies a `judge_server_url` (`host:port` of an already-running
`vllm serve` endpoint) and a `judge_model` name. The judge is queried via the
OpenAI **Chat Completions** API (`{judge_server_url}/v1/chat/completions`),
which is the surface a stock `vllm serve` exposes.

All math-verify / verdict / data-schema logic is inherited unchanged from
[`resources_servers/math_with_judge_original`](../math_with_judge_original),
which is the original Gym-managed-judge variant.

## How it differs from `math_with_judge_original`

| Aspect | `math_with_judge_original` | `math_with_judge` (this) |
|--------|----------------------------|--------------------------|
| Judge hosting | NeMo-Gym manages a `responses_api_models` vLLM server | Externally hosted; YAML supplies `judge_server_url` |
| Judge protocol | `/v1/responses` (Responses API) via `ServerClient` | `/v1/chat/completions` (native vLLM) via direct HTTP |
| Config fields | `judge_model_server` (a `ModelServerRef`) | `judge_server_url` + `judge_model` (both mandated, no defaults) |
| Init validation | (none — Gym spins up the judge) | `judge_server_url` normalized + `judge_model` checked against `/v1/models` |
| Math-verify / verdict / data schema | — | **Identical** (inherited from `math_with_judge_original`) |

Everything else — `[[A=B]]`/`[[A!=B]]` verdict extraction, `should_use_judge` gate
(judge only fires when `should_use_judge=True` AND library reward ≤ 0.5),
two-ordered-pass positional-bias elimination, request/response models — is
inherited unchanged.

## Data compatibility

Any dataset compatible with `math_with_judge_simple_agent` (the original variant
under `math_with_judge_original/`) works here **unchanged** — the
`agent_ref.name` is the same (`math_with_judge_simple_agent`). See
`data/README.md` for details.

## Configuration

```yaml
math_with_judge:
  resources_servers:
    math_with_judge:
      entrypoint: app.py

      # MANDATED (no defaults):
      judge_server_url: "0.0.0.0:8000"   # host:port of the external vLLM judge
      judge_model: "Qwen/Qwen3-30B-A3B-Instruct-2507"

      judge_responses_create_params:
        input: []
        max_output_tokens: 8192
        temperature: 0.7
        top_p: 0.8

      should_use_judge: true
```

### NeMo-RL integration

To use this from an NRL GRPO config, add the config path and override
`judge_server_url`/`judge_model`:

```yaml
env:
  nemo_gym:
    config_paths:
      - resources_servers/math_with_judge/configs/math_with_judge.yaml
    math_with_judge:
      resources_servers:
        math_with_judge:
          judge_server_url: "0.0.0.0:8000"
          judge_model: "Qwen/Qwen3-30B-A3B-Instruct-2507"
          judge_responses_create_params:
            max_output_tokens: 8192
            temperature: ${policy.generation.temperature}
            top_p: ${policy.generation.top_p}
          should_use_judge: true
```

## Testing

```bash
ng_test +entrypoint=resources_servers/math_with_judge
```

Or directly:

```bash
cd 3rdparty/Gym-workspace/Gym && conda run -n trashrepo_v2 env PYTHONPATH=. \
  python -m pytest resources_servers/math_with_judge/tests/ -v --timeout=60
```

Tests cover (no network required):
- `judge_server_url` normalization (`0.0.0.0:8000`, full URLs, path stripping, empty rejection)
- chat-completions payload building (`max_output_tokens` → `max_tokens`, temperature/top_p)
- chat-completion text extraction (string and list content)
- minimal `NeMoGymResponse` construction wrapping chat-completions output
- config required-field enforcement (`judge_server_url` / `judge_model` / `judge_responses_create_params`)
- inherited library verifier and math delimiter stripping

## File Structure

```
math_with_judge/
├── app.py                          # Server subclass + chat-completions judge
├── requirements.txt                # -e nemo-gym[dev] @ ../../, math-verify
├── README.md                       # This file
├── configs/
│   └── math_with_judge.yaml        # Server + agent configuration
├── data/
│   ├── example.jsonl               # Example data (committed)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_app.py
```
