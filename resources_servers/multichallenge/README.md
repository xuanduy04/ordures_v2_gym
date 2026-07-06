# MultiChallenge Environment

Evaluates model responses on the **MultiChallenge** benchmark using an LLM judge
whose hosting is **not managed by NeMo-Gym**. The YAML config supplies a
`judge_server_url` (`host:port` of an already-running `vllm serve` endpoint) and
a `judge_model` name. The judge is queried via the OpenAI **Chat Completions**
API (`{judge_server_url}/v1/chat/completions`), which is the surface a stock
`vllm serve` exposes.

All rubric / aggregation / verdict / data-schema logic is inherited unchanged
from [`resources_servers/multichallenge_original`](../multichallenge_original),
which is the original Gym-managed-judge variant.

## Quick Start

```bash
# 1. Run unit tests
ng_test +entrypoint=resources_servers/multichallenge

# 2. Start servers (in terminal 1)
#    A separate `vllm serve` judge must already be running at judge_server_url.
config_paths="resources_servers/multichallenge/configs/multichallenge.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# 3. Collect rollouts on example data (in terminal 2)
ng_collect_rollouts \
  +agent_name=multichallenge_simple_agent \
  +input_jsonl_fpath=resources_servers/multichallenge/data/example.jsonl \
  +output_jsonl_fpath=/tmp/multichallenge_rollouts.jsonl
```

## How it differs from `multichallenge_original`

| Aspect | `multichallenge_original` | `multichallenge` (this) |
|--------|---------------------------|-------------------------|
| Judge hosting | NeMo-Gym manages a `responses_api_models` vLLM server | Externally hosted; YAML supplies `judge_server_url` |
| Judge protocol | `/v1/responses` (Responses API) via `ServerClient` | `/v1/chat/completions` (native vLLM) via direct HTTP |
| Config fields | `judge_model_server` (a `ModelServerRef`) | `judge_server_url` + `judge_model` (both mandated, no defaults) |
| Init validation | (none — Gym spins up the judge) | `judge_server_url` normalized + `judge_model` checked against `/v1/models` |
| Rubric / aggregation / verdict / data schema | — | **Identical** (inherited from `multichallenge_original`) |

Everything else — prompt template, `[[YES]]`/`[[NO]]` verdict extraction,
`aggregation_mode`, `parallel_evaluation`, request/response models, the `verify()`
flow — is inherited unchanged.

## Data compatibility

Any dataset compatible with `multichallenge_simple_agent` (the original variant
under `multichallenge_original/`) works here **unchanged** — the
`agent_ref.name` is the same (`multichallenge_simple_agent`). No field needs to
change (assuming valid YAML config settings for `judge_model` and
`judge_server_url`). See `data/README.md` for details.

## Configuration

```yaml
multichallenge:
  resources_servers:
    multichallenge:
      entrypoint: app.py

      # MANDATED (no defaults):
      judge_server_url: "0.0.0.0:8000"   # host:port of the external vLLM judge
      judge_model: "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

      # max_output_tokens maps to max_tokens in the chat-completions payload.
      judge_responses_create_params:
        input: []
        max_output_tokens: 8192
        temperature: 0.7
        top_p: 0.8

      aggregation_mode: mean
      parallel_evaluation: true
```

### NeMo-RL integration

To use this from an NRL GRPO config, add the config path and override
`judge_server_url`/`judge_model` (pointing at the GenRM-compatible judge model,
the same one used by e.g. `examples/configs/super/stage1_rlvr.yaml`'s
`nl2bash_judge_model`, but hosted externally):

```yaml
env:
  nemo_gym:
    config_paths:
      - resources_servers/multichallenge/configs/multichallenge.yaml
    multichallenge:
      resources_servers:
        multichallenge:
          judge_server_url: "0.0.0.0:8000"
          judge_model: "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
          judge_responses_create_params:
            max_output_tokens: 8192
            temperature: ${policy.generation.temperature}
            top_p: ${policy.generation.top_p}
```

## Testing

```bash
ng_test +entrypoint=resources_servers/multichallenge
```

Tests cover (no network required):
- `judge_server_url` normalization (`0.0.0.0:8000`, full URLs, path stripping, empty rejection)
- chat-completions payload building (`max_output_tokens` → `max_tokens`, temperature/top_p)
- chat-completion text extraction (string and list content)
- config required-field enforcement (`judge_server_url` / `judge_model` / `judge_responses_create_params`)
- inherited aggregation logic (mean / all)

## File Structure

```
multichallenge/
├── app.py                          # Server subclass + chat-completions judge
├── requirements.txt                # -e nemo-gym[dev] @ ../../
├── README.md                       # This file
├── configs/
│   └── multichallenge.yaml         # Server + agent configuration
├── data/
│   ├── example.jsonl               # Example data (committed)
│   ├── example_rollouts.jsonl      # Example rollouts (committed)
│   ├── example_metrics.json        # Example metrics (committed)
│   ├── advanced.jsonl              # Full split (generated, gitignored)
│   ├── vanilla.jsonl               # Full split (generated, gitignored)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_multichallenge.py
```

## API Endpoints

- `POST /verify` — Evaluate a model response against the rubric (identical
  request/response schema to `multichallenge_original`).
- `POST /seed_session` — Initialize a new session.
