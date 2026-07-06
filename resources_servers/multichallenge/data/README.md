# MultiChallenge Data

This directory holds JSONL data for the `multichallenge_simple_agent` agent.

## Compatibility with `multichallenge_original`

The data schema is **identical** to that of
`resources_servers/multichallenge_original/data/`. Any dataset that is
compatible with `multichallenge_simple_agent` (the original, Gym-managed-judge
variant under `multichallenge_original/`) can be reused here **verbatim** —
the `agent_ref.name` is the same (`multichallenge_simple_agent`).

No other field needs to change (assuming valid YAML config settings for
`judge_model` and `judge_server_url`).

## Files

- `example.jsonl` — example rows (committed), copied from
  `multichallenge_original/data/example.jsonl`.
- `example_rollouts.jsonl` — example rollouts (committed).
- `example_metrics.json` — example metrics (committed).
- `advanced.jsonl` / `vanilla.jsonl` — full splits (generated, gitignored).
  Produce them with `resources_servers/multichallenge_original/dataset_preprocess.py`.
