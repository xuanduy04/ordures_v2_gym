# MultiChallenge Outsource Environment

Evaluates model responses on the **MultiChallenge** benchmark using an LLM judge —
identical to [`resources_servers/multichallenge`](../multichallenge) except that
the LLM-judge is **not managed by NeMo-Gym**.

Instead, the YAML config supplies a `judge_link` (`host:port` of an already-running
`vllm serve` endpoint) and a `judge_model` name. The judge is queried via the
OpenAI **Chat Completions** API (`{judge_link}/v1/chat/completions`), which is the
surface which a stock `vllm serve` exposes.

## Quick Start

```bash
# 1. Run unit tests
ng_test +entrypoint=resources_servers/multichallenge_outsource

# 2. Start servers (in terminal 1)
#    A separate `vllm serve` judge must already be running at judge_link.
config_paths="resources_servers/multichallenge_outsource/configs/multichallenge_outsource.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# 3. Collect rollouts on example data (in terminal 2)
ng_collect_rollouts \
  +agent_name=multichallenge_simple_agent_outsource \
  +input_jsonl_fpath=resources_servers/multichallenge_outsource/data/example.jsonl \
  +output_jsonl_fpath=/tmp/multichallenge_outsource_rollouts.jsonl
```

## How it differs from `multichallenge`

| Aspect | `multichallenge` | `multichallenge_outsource` |
|--------|------------------|----------------------------|
| Judge hosting | NeMo-Gym manages a `responses_api_models` vLLM server | Externally hosted; YAML supplies `judge_link` |
| Judge protocol | `/v1/responses` (Responses API) via `ServerClient` | `/v1/chat/completions` (native vLLM) via direct HTTP |
| Config fields | `judge_model_server` (a `ModelServerRef`) | `judge_link` + `judge_model` (both mandated, no defaults) |
| Init validation | (none — Gym spins up the judge) | `judge_link` normalized + `judge_model` checked against `/v1/models` |
| Rubric / aggregation / verdict / data schema | — | **Identical** (inherited from `MultiChallengeServer`) |

Everything else — prompt template, `[[YES]]`/`[[NO]]` verdict extraction,
`aggregation_mode`, `parallel_evaluation`, request/response models, the `verify()`
flow — is inherited unchanged from `MultiChallengeServer`.

## Data compatibility

Any dataset compatible with `multichallenge_simple_agent` can be reused **verbatim
except** the `agent_ref.name` field, which must change from
`multichallenge_simple_agent` to `multichallenge_simple_agent_outsource`:

```bash
sed 's/multichallenge_simple_agent/multichallenge_simple_agent_outsource/g' \
  resources_servers/multichallenge/data/advanced.jsonl \
  > resources_servers/multichallenge_outsource/data/advanced.jsonl
```

No other field needs to change (assuming valid YAML config settings for
`judge_model` and `judge_link`). See `data/README.md` for details.

## Configuration

```yaml
multichallenge_outsource:
  resources_servers:
    multichallenge_outsource:
      entrypoint: app.py

      # MANDATED (no defaults):
      judge_link: "0.0.0.0:8000"   # host:port of the external vLLM judge
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
`judge_link`/`judge_model` (pointing at the GenRM-compatible judge model, the
same one used by e.g. `examples/configs/super/stage1_rlvr.yaml`'s
`nl2bash_judge_model`, but hosted externally):

```yaml
env:
  nemo_gym:
    config_paths:
      - resources_servers/multichallenge_outsource/configs/multichallenge_outsource.yaml
    multichallenge_outsource:
      resources_servers:
        multichallenge_outsource:
          judge_link: "0.0.0.0:8000"
          judge_model: "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
```

## Testing

```bash
ng_test +entrypoint=resources_servers/multichallenge_outsource
```

Tests cover (no network required):
- `judge_link` normalization (`0.0.0.0:8000`, full URLs, path stripping, empty rejection)
- chat-completions payload building (`max_output_tokens` → `max_tokens`, temperature/top_p)
- chat-completion text extraction (string and list content)
- config required-field enforcement (`judge_link` / `judge_model` / `judge_responses_create_params`)
- inherited aggregation logic (mean / all)

## File Structure

```
multichallenge_outsource/
├── app.py                              # Server subclass + chat-completions judge
├── requirements.txt                    # -e nemo-gym[dev] @ ../../
├── README.md                           # This file
├── configs/
│   └── multichallenge_outsource.yaml   # Server + agent configuration
├── data/
│   ├── example.jsonl                   # Example data (agent_ref.name swapped)
│   ├── advanced.jsonl                  # Full split (generated, gitignored)
│   ├── vanilla.jsonl                   # Full split (generated, gitignored)
│   ├── .gitignore
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_multichallenge_outsource.py
```

## API Endpoints

- `POST /verify` — Evaluate a model response against the rubric (identical
  request/response schema to `multichallenge`).
- `POST /seed_session` — Initialize a new session.
