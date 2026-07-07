# Math With Judge Data

This directory holds JSONL data for the `math_with_judge_simple_agent` agent.

## Compatibility with `math_with_judge_original`

The data schema is **identical** to that of
`resources_servers/math_with_judge_original/data/`. Any dataset that is
compatible with `math_with_judge_simple_agent` (the original, Gym-managed-judge
variant under `math_with_judge_original/`) can be reused here **verbatim** —
the `agent_ref.name` is the same (`math_with_judge_simple_agent`).

No other field needs to change (assuming valid YAML config settings for
`judge_model` and `judge_server_url`).

## Files

- `example.jsonl` — example rows (committed), copied from
  `math_with_judge_original/data/example.jsonl`.
- `*.jsonl` — full splits (generated, gitignored).
