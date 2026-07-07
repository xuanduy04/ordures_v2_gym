# GenRM Compare Data

This directory holds JSONL data for the `genrm_simple_agent` and
`genrm_simple_agent_reasoning_off` agents.

## Compatibility with `genrm_compare_original`

The data schema is **identical** to that of
`resources_servers/genrm_compare_original/data/`. Any dataset that is
compatible with `genrm_simple_agent` (the original, Gym-managed GenRM judge
variant under `genrm_compare_original/`) can be reused here **verbatim** —
the `agent_ref.name` is the same (`genrm_simple_agent`).

No other field needs to change (assuming valid YAML config settings for
`genrm_model` and `genrm_server_url`).

## Files

- `example.jsonl` — example rows (committed), copied from
  `genrm_compare_original/data/example.jsonl`.
