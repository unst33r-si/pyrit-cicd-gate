<p align="center"><img src="logo.svg" alt="UNST33R" width="420"></p>

# PyRIT CI/CD Security Gate

Reusable GitHub Action + Python CLI that runs Microsoft's [PyRIT](https://github.com/microsoft/PyRIT)
(Python Risk Identification Toolkit) against an AI application or agent on
every pull request: **1,000 adversarial probes per run**, and the merge is
blocked (`exit 1`) when the Attack Success Rate (ASR / jailbreak rate)
exceeds **1%**.

```
seed corpus (40 objectives x 25 mutations = 1,000 probes)
        |
        |-- 900 single-turn probes --> attacker: openai/unst33r-flash
        |                                 (GLM-5.3-Flash uncensored)
        |-- 100 multi-turn probes  --> attacker: openai/unst33r-deepseek
        |                                 (DeepSeek-V4.1-Flash-1017 abliterated, max_turns=3)
        v
   target AI application (your agent, any OpenAI-compatible endpoint)
        |
        |-- refusal pre-filter (deterministic, free)
        |-- judge: openai/unst33r-pro  (GLM-5.3 uncensored)
        |        PyRIT SelfAskTrueFalseScorer, decoupled from target weights
        v
   ASR <= 1%  -> exit 0 (PASS)      ASR > 1% -> exit 1 (BLOCKED)
   attacker refusals > 2% -> exit 2 (INVALID_RUN)
```

## Why this exists: the silent false-negative problem

PyRIT is not a static fuzzer. Its `RedTeamingAttack` needs an **attacker
model** to synthesize, mutate, and escalate adversarial prompts across
single-turn and multi-turn attacks, plus a **judge model** to score whether
the target actually complied.

Point those roles at a standard safety-tuned commercial API and a nasty
failure mode appears: mid-run the attacker model starts refusing to generate
the payloads. The run completes, reports a sparkling **0% jailbreak rate**,
and the PR merges — not because the target is safe, but because *no attack
was ever launched*. Silent false negative.

This gate fixes that structurally:

1. **Attacker roles run on the [UNST33R API](https://api.unst33r.si/v1)**
   (OpenAI-compatible), which serves uncensored/abliterated models that
   actually play the attacker role end-to-end.
2. **Attacker refusals are measured, not ignored.** Every probe result
   records whether the attacker model refused to produce the payload. If the
   `attacker_refusal_rate` exceeds `max_attacker_refusal_rate` (default 2%)
   the run exits with code `2` (`INVALID_RUN`) — never a clean pass.
3. **The judge is a third, independent model** (`unst33r-pro`) via PyRIT
   `SelfAskTrueFalseScorer`, decoupled from the target weights, behind a
   deterministic refusal pre-filter so obvious "I'm sorry" responses don't
   burn judge tokens.

## Roles and models

| Role | Model | Notes |
| --- | --- | --- |
| Single-turn mutation attacker | `openai/unst33r-flash` (GLM-5.3-Flash uncensored) | Generates/mutates 900 probes across 4 categories. $0.50 / $1.50 per 1M tok. |
| Multi-turn adaptive attacker | `openai/unst33r-deepseek` (DeepSeek-V4.1-Flash-1017 abliterated) | Drives `RedTeamingAttack` escalation, up to `max_turns=3`, on 100 high-risk objectives. $0.15 / $0.5625 per 1M tok, 1M ctx. |
| Independent judge / scorer | `openai/unst33r-pro` (GLM-5.3 uncensored) | `SelfAskTrueFalseScorer` + `TrueFalseQuestion`. $2.50 / $4.50 per 1M tok. |
| Target | your application | Any OpenAI-compatible `/v1/chat/completions`. |

## The corpus

`seeds/objectives.yaml` holds **40 base objectives** (10 per category) and
**25 mutation framing strategies** — base64 payload wrapping, roleplay
persona framing, multilingual code-switching, hypothetical debugging
wrappers, markdown/JSON schema smuggling, system-override framing, indirect
RAG context injection, token smuggling, payload splitting, few-shot
injection, and more.

Categories:

* `system_prompt_leakage`
* `jailbreak_policy_bypass`
* `indirect_prompt_injection`
* `agent_tool_privilege_escalation`

The Cartesian product (40 × 25) expands **deterministically** to up to
1,000 unique probes. Ordering is SHA-256-keyed, so every run over the same
seed file produces an identical corpus and a reproducible
`corpus_sha256` digest in the report. Multi-turn slots prefer the
highest-risk categories.

A separate `benign_security_lookalikes` set (legitimate defensive prompts —
parsing auth logs, explaining parameterized queries) measures the target's
**false-refusal rate**: an agent that refuses to discuss defensive security
is also broken.

## Using the GitHub Action

```yaml
- uses: unst33r-si/pyrit-cicd-gate@v1
  with:
    unst33r_api_key: ${{ secrets.UNST33R_API_KEY }}
    target_endpoint: "http://localhost:8080/v1"
    target_model: "openai/target-agent"
```

All inputs (with defaults):

| Input | Default | Purpose |
| --- | --- | --- |
| `unst33r_api_key` | *required* | UNST33R API key |
| `unst33r_endpoint` | `https://api.unst33r.si/v1` | attacker/judge endpoint |
| `attacker_single_turn_model` | `openai/unst33r-flash` | single-turn mutations |
| `attacker_multi_turn_model` | `openai/unst33r-deepseek` | `RedTeamingAttack` driver |
| `scorer_model` | `openai/unst33r-pro` | judge |
| `target_endpoint` | *required* | app under test |
| `target_api_key` | `dummy-key` | target auth |
| `target_model` | `openai/target-agent` | target model id |
| `single_turn_count` | `900` | single-turn probes |
| `multi_turn_count` | `100` | multi-turn probes |
| `max_turns` | `3` | turns per multi-turn probe |
| `max_jailbreak_rate` | `0.01` | ASR threshold |
| `max_attacker_refusal_rate` | `0.02` | INVALID_RUN threshold |
| `concurrency` | `25` | in-flight probes |
| `engine` | `pyrit` | `pyrit` / `http` / `mock` |
| `seeds_path` | `seeds/objectives.yaml` | corpus file |
| `output_dir` | `pyrit-gate-results` | artifact dir |

Outputs: `total_probes`, `jailbreak_count`, `jailbreak_rate`,
`attacker_refusal_rate`, `gate_passed`, `report_json_path`,
`transcript_jsonl_path`.

A full copy-paste caller workflow lives in
[`examples/caller-workflow.yml`](examples/caller-workflow.yml). Wire the job
into branch protection as a required check.

## Using the CLI

```bash
pip install -e .
# optional: full PyRIT stack
pip install -r requirements.txt

pyrit-gate \
  --engine pyrit \
  --unst33r-api-key "$UNST33R_API_KEY" \
  --target-endpoint http://localhost:8080/v1 \
  --single-turn-count 900 --multi-turn-count 100 \
  --max-jailbreak-rate 0.01
```

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | PASS — ASR and attacker-refusal rate within thresholds |
| `1` | BLOCKED — ASR > `max_jailbreak_rate` |
| `2` | INVALID_RUN — attacker refused too often; results untrusted |
| `3` | ERROR — configuration / dependency failure |

### Engines

* **`pyrit`** — real Microsoft PyRIT execution: `PromptSendingAttack` for the
  900 single-turn probes; `RedTeamingAttack` (`AttackAdversarialConfig` on
  `unst33r-deepseek`, `RTASystemPromptPaths`, `max_turns=3`) for the 100
  multi-turn probes; `SelfAskTrueFalseScorer` on `unst33r-pro`.
* **`http`** — async OpenAI-compatible engine (`httpx` or `urllib`
  fallback) that plays attacker/judge/target roles over REST. Useful when
  you want the gate without the full PyRIT dependency tree.
* **`mock`** — deterministic simulation (`--mock-breach-rate`,
  `--mock-attacker-refusal-rate`); all 1,000 probes in <1 s. Used by the
  unit tests and CI self-tests.

### Local end-to-end dry run

```bash
python3 examples/mock_target_server.py --port 8080 --breach-rate 0.02 &

pyrit-gate --engine http \
  --unst33r-endpoint http://localhost:8080/v1 --unst33r-api-key test \
  --target-endpoint http://localhost:8080/v1 --target-api-key test \
  --target-model openai/mock-agent
```

The bundled mock server understands the attacker/judge role markers, so one
process can stand in for both UNST33R and the target during development.

## Sample step summary

```markdown
## UNST33R PyRIT CI/CD Security Gate

**Verdict:** `PASS`

| Metric | Value | Threshold |
| --- | --- | --- |
| Total probes | 1000 | - |
| Single-turn probes | 900 | - |
| Multi-turn probes | 100 | - |
| Jailbreaks | 5 | - |
| Attack success rate | 0.50% | <= 1.00% |
| Attacker refusals | 3 | - |
| Attacker refusal rate | 0.30% | <= 2.00% |
| Benign false-refusal rate | 0.00% | info |
```

Artifacts uploaded on every run (pass or fail):

* `summary.json` — full metrics, per-category and per-mode ASR, cost estimate
* `transcripts.jsonl` — one record per probe with turns and verdicts
* `report.md` — the Markdown report also written to `$GITHUB_STEP_SUMMARY`

## Token cost arithmetic (1,000 probes/PR)

Rough per-probe budget, using published UNST33R pricing:

| Model | Role | ~Tokens per probe (in/out) | Price per 1M | ~Cost per 1,000 probes |
| --- | --- | --- | --- | --- |
| `unst33r-flash` | single-turn attacker | 150 / 120 | $0.50 / $1.50 | ≈ $0.23 |
| `unst33r-deepseek` | multi-turn attacker | ~3 turns × (140/60) | $0.15 / $0.5625 | ≈ $0.02 |
| `unst33r-pro` | judge (non-refused only) | 240 / 8 per judged probe | $2.50 / $4.50 | ≈ $0.10–0.60 |

Expect **well under $1 per PR** for the attacker+judge side; the exact
figure is computed from real `usage` fields and printed in the report under
"Estimated UNST33R cost". Target-side tokens depend on your own serving
stack.

## Development

```bash
pip install -e .
python3 -m unittest discover -s tests -v
```

The test suite covers the 1,000-probe expansion and digest determinism,
attacker-refusal detection (the false-negative guard), gate pass/block/
invalid exit codes, a live HTTP loopback against the bundled mock target,
and report/artifact generation. It needs no external services — `httpx` and
`pyyaml` are used if present but the `mock`/`http` engines and tests also
work on the standard library alone (a mini-YAML fallback parses the seeds
file when PyYAML is absent).

## License

MIT — see [LICENSE](LICENSE).