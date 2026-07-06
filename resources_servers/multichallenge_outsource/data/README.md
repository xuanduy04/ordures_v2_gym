# MultiChallenge Outsource Data

This directory holds JSONL data for the `multichallenge_simple_agent_outsource`
agent.

## Compatibility with `multichallenge_simple_agent`

The data schema is **identical** to that of
`resources_servers/multichallenge/data/`. Any dataset that is compatible with
`multichallenge_simple_agent` can be reused here **verbatim except** for the
`agent_ref.name` field, which must be changed from `multichallenge_simple_agent`
to `multichallenge_simple_agent_outsource`:

```bash
sed 's/multichallenge_simple_agent/multichallenge_simple_agent_outsource/g' \
  resources_servers/multichallenge/data/advanced.jsonl \
  > resources_servers/multichallenge_outsource/data/advanced.jsonl
```

No other field needs to change (assuming valid YAML config settings for
`judge_model` and `judge_server_url`).

## Files

- `example.jsonl` — 5 example rows (committed), copied from
  `multichallenge/data/example.jsonl` with `agent_ref.name` swapped.
- `advanced.jsonl` / `vanilla.jsonl` — full splits (generated, gitignored).
  Produce them with the `sed` command above or by re-running
  `resources_servers/multichallenge/dataset_preprocess.py` after swapping the
  agent name in its hardcoded `name` field.
