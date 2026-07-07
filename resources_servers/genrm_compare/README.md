# GenRM Compare Environment

Compares multiple candidate responses using a **GenRM (Generative Reward Model)**
whose hosting is **not managed by NeMo-Gym**. The YAML config supplies a
`genrm_server_url` (`host:port` of an already-running GenRM model endpoint) and
a `genrm_model` name. The GenRM model is queried via the OpenAI **Chat
Completions** API (`{genrm_server_url}/v1/chat/completions`), which is the
surface a stock `vllm serve` exposes.

All pairwise comparison / aggregation / GenRM output parsing / data-schema logic
is inherited unchanged from
[`resources_servers/genrm_compare_original`](../genrm_compare_original),
which is the original Gym-managed GenRM model variant.

## Quick Start

```bash
# 1. Run unit tests
ng_test +entrypoint=resources_servers/genrm_compare

# 2. A separate GenRM model endpoint must already be running at genrm_server_url.
```

## How it differs from `genrm_compare_original`

| Aspect | `genrm_compare_original` | `genrm_compare` (this) |
|--------|--------------------------|------------------------|
| GenRM hosting | NeMo-Gym manages a `responses_api_models` vLLM server | Externally hosted; YAML supplies `genrm_server_url` |
| GenRM protocol | `/v1/responses` (Responses API) via `ServerClient` | `/v1/chat/completions` (native vLLM) via direct HTTP |
| Config fields | `genrm_model_server` (a `ModelServerRef`) | `genrm_server_url` + `genrm_model` (both mandated, no defaults) |
| Init validation | (none — Gym spins up the GenRM model) | `genrm_server_url` normalized + `genrm_model` checked against `/v1/models` |
| Comparison / aggregation / parse / data schema | — | **Identical** (inherited from `genrm_compare_original`) |

Everything else — `comparison_strategy`, `num_judges_per_comparison`,
`aggregator_method`, length/style bonuses, `parse_genrm_output`,
`generate_comparison_pairs`, the `compare()` and `verify()` endpoints, and
the `GenRMCompareRequest` / `GenRMCompareResponse` models — is inherited
unchanged.

## Data compatibility

Any dataset compatible with `genrm_simple_agent` (the original variant under
`genrm_compare_original/`) works here **unchanged** — the `agent_ref.name`
is the same (`genrm_simple_agent`). See `data/README.md` for details.

## Configuration

```yaml
genrm_compare:
  resources_servers:
    genrm_compare:
      entrypoint: app.py

      # MANDATED (no defaults):
      genrm_server_url: "0.0.0.0:8000"
      genrm_model: "nvidia/Qwen3-Nemotron-235B-A22B-GenRM"

      # max_output_tokens maps to max_tokens in the chat-completions payload.
      genrm_responses_create_params:
        input: []
        max_output_tokens: 16384
        temperature: 0.6
        top_p: 0.95

      comparison_strategy: circular
      num_judges_per_comparison: 1
```

### Special roles

The GenRM model's chat template expects special roles `response_1` and
`response_2` (and optionally `principle`). These are preserved in the
chat-completions `messages` array — they are NOT collapsed to
`user`/`assistant`. The external GenRM model must support these roles.

### Expected GenRM output format

The GenRM model should output JSON:

```json
{"score_1": 4, "score_2": 3, "ranking": 2}
```

### NeMo-RL integration

To use this from an NRL GRPO config, add the config path and override
`genrm_server_url`/`genrm_model`:

```yaml
env:
  nemo_gym:
    config_paths:
      - resources_servers/genrm_compare/configs/genrm_compare.yaml
    genrm_compare:
      resources_servers:
        genrm_compare:
          genrm_server_url: "0.0.0.0:8000"
          genrm_model: "nvidia/Qwen3-Nemotron-235B-A22B-GenRM"
          genrm_responses_create_params:
            max_output_tokens: 16384
            temperature: ${policy.generation.temperature}
            top_p: ${policy.generation.top_p}
```

## File Structure

```
genrm_compare/
├── app.py                          # Server subclass + chat-completions GenRM call
├── requirements.txt                # -e nemo-gym[dev] @ ../../
├── README.md                       # This file
├── configs/
│   └── genrm_compare.yaml         # Server + agent configuration
├── data/
│   ├── example.jsonl              # Example data (committed)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_app.py
```

## API Endpoints

- `POST /compare` — Compare multiple candidate responses via GenRM pairwise comparison (identical request/response schema to `genrm_compare_original`)
- `POST /verify` — Stub endpoint returning default score (inherited)

## Testing

```bash
ng_test +entrypoint=resources_servers/genrm_compare
```

Tests cover (no network required):
- Config required-field enforcement (`genrm_server_url` / `genrm_model` / `genrm_responses_create_params`)
- `genrm_model_server` field absent (proves external hosting)
- Default name (`genrm_compare`)
- Inherited comparison logic (`compare()` single-response default, `verify()` stub)
