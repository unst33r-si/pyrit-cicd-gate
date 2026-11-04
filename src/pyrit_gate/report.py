"""Reporting: summary.json, transcripts.jsonl, Markdown step summary."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from pyrit_gate.config import GateConfig, MODEL_PRICING, pricing_key
from pyrit_gate.runner import ProbeResult, RunSummary

TOKEN_ROLE_TO_MODEL = {
    "attacker_single_turn_model": "attacker_single_turn_model",
    "attacker_multi_turn_model": "attacker_multi_turn_model",
    "scorer_model": "scorer_model",
    "target_model": "target_model",
}

VERDICTS = {0: "PASS", 1: "BLOCKED", 2: "INVALID_RUN", 3: "ERROR"}


def compute_exit_code(jailbreak_rate: float, attacker_refusal_rate: float,
                      config: GateConfig) -> int:
    """Exit-code contract.

    ``2`` (INVALID_RUN) takes precedence: a run where the attacker model
    refused to generate probes cannot be trusted either way.
    """
    if attacker_refusal_rate > config.max_attacker_refusal_rate:
        return 2
    if jailbreak_rate > config.max_jailbreak_rate:
        return 1
    return 0


def _rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def _cost_estimate(token_totals: Dict[str, Dict[str, int]],
                   config: GateConfig) -> Dict[str, Any]:
    """USD estimate per model from recorded (or estimated) token usage."""
    role_models = {
        "attacker_single_turn_model": config.attacker_single_turn_model,
        "attacker_multi_turn_model": config.attacker_multi_turn_model,
        "scorer_model": config.scorer_model,
        "target_model": config.target_model,
    }
    models: Dict[str, Dict[str, Any]] = {}
    total = 0.0
    for role, totals in token_totals.items():
        model = role_models.get(role, config.target_model)
        key = pricing_key(model)
        in_rate, out_rate = MODEL_PRICING.get(key, (0.0, 0.0))
        cost = (totals["in"] / 1e6) * in_rate + (totals["out"] / 1e6) * out_rate
        models[key] = {
            "model": model,
            "prompt_tokens": totals["in"],
            "completion_tokens": totals["out"],
            "input_price_per_1m": in_rate,
            "output_price_per_1m": out_rate,
            "estimated_usd": round(cost, 6),
        }
        total += cost
    return {"per_model": models, "total_estimated_usd": round(total, 6)}


def build_summary(config: GateConfig, run: RunSummary) -> Dict[str, Any]:
    results = run.results
    benign = run.benign_results
    total = len(results)
    jailbreaks = [r for r in results if r.jailbroken]
    attacker_refusals = [r for r in results if r.attacker_refused]
    errors = [r for r in results if r.error]

    per_category: Dict[str, Dict[str, Any]] = {}
    for r in results:
        slot = per_category.setdefault(
            r.category, {"probes": 0, "jailbreaks": 0, "attacker_refusals": 0})
        slot["probes"] += 1
        slot["jailbreaks"] += int(r.jailbroken)
        slot["attacker_refusals"] += int(r.attacker_refused)
    for slot in per_category.values():
        slot["asr"] = round(_rate(slot["jailbreaks"], slot["probes"]), 6)

    per_mode: Dict[str, Dict[str, Any]] = {}
    for r in results:
        slot = per_mode.setdefault(
            r.mode, {"probes": 0, "jailbreaks": 0})
        slot["probes"] += 1
        slot["jailbreaks"] += int(r.jailbroken)
    for slot in per_mode.values():
        slot["asr"] = round(_rate(slot["jailbreaks"], slot["probes"]), 6)

    token_totals: Dict[str, Dict[str, int]] = {}
    for r in results + benign:
        for role, t in r.tokens.items():
            slot = token_totals.setdefault(role, {"in": 0, "out": 0})
            slot["in"] += t.get("in", 0)
            slot["out"] += t.get("out", 0)

    jailbreak_rate = _rate(len(jailbreaks), total)
    attacker_refusal_rate = _rate(len(attacker_refusals), total)
    benign_refused = sum(1 for r in benign if r.target_refused)
    exit_code = compute_exit_code(jailbreak_rate, attacker_refusal_rate, config)

    return {
        "gate": "pyrit-cicd-gate",
        "engine": run.engine,
        "verdict": VERDICTS.get(exit_code, "ERROR"),
        "exit_code": exit_code,
        "total_probes": total,
        "single_turn_probes": sum(1 for r in results if r.mode == "single_turn"),
        "multi_turn_probes": sum(1 for r in results if r.mode == "multi_turn"),
        "jailbreak_count": len(jailbreaks),
        "jailbreak_rate": round(jailbreak_rate, 6),
        "max_jailbreak_rate": config.max_jailbreak_rate,
        "attacker_refusals": len(attacker_refusals),
        "attacker_refusal_rate": round(attacker_refusal_rate, 6),
        "max_attacker_refusal_rate": config.max_attacker_refusal_rate,
        "execution_errors": len(errors),
        "benign_probes": len(benign),
        "benign_false_refusals": benign_refused,
        "benign_false_refusal_rate": round(_rate(benign_refused, len(benign)), 6),
        "per_category": per_category,
        "per_mode": per_mode,
        "token_totals": token_totals,
        "cost_estimate": _cost_estimate(token_totals, config),
        "corpus_sha256": run.corpus_sha256,
        "seeds_path": config.seeds_path,
        "elapsed_s": round(run.elapsed_s, 3),
        "models": {
            "attacker_single_turn": config.attacker_single_turn_model,
            "attacker_multi_turn": config.attacker_multi_turn_model,
            "scorer": config.scorer_model,
            "target": config.target_model,
        },
        "thresholds": {
            "max_jailbreak_rate": config.max_jailbreak_rate,
            "max_attacker_refusal_rate": config.max_attacker_refusal_rate,
        },
    }


def _result_record(r: ProbeResult) -> Dict[str, Any]:
    return {
        "probe_id": r.probe_id,
        "mode": r.mode,
        "category": r.category,
        "objective": r.objective,
        "mutation": r.mutation,
        "attack_prompt": r.attack_prompt,
        "attacker_refused": r.attacker_refused,
        "target_response": r.target_response,
        "target_refused": r.target_refused,
        "judge_verdict": r.judge_verdict,
        "jailbroken": r.jailbroken,
        "turns": r.turns,
        "latency_s": round(r.latency_s, 4),
        "tokens": r.tokens,
        "error": r.error,
    }


def write_outputs(summary: Dict[str, Any], run: RunSummary,
                  output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    transcript_path = os.path.join(output_dir, "transcripts.jsonl")
    report_path = os.path.join(output_dir, "report.md")

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    with open(transcript_path, "w", encoding="utf-8") as fh:
        for r in run.results:
            fh.write(json.dumps(_result_record(r), sort_keys=True) + "\n")
        for r in run.benign_results:
            fh.write(json.dumps(_result_record(r), sort_keys=True) + "\n")

    markdown = format_markdown(summary)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)

    return {
        "summary_json": summary_path,
        "transcript_jsonl": transcript_path,
        "report_md": report_path,
    }


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def format_markdown(summary: Dict[str, Any]) -> str:
    verdict = summary["verdict"]
    badge = {
        "PASS": "PASS",
        "BLOCKED": "BLOCKED - ASR above threshold",
        "INVALID_RUN": "INVALID RUN - attacker model refused too often",
    }.get(verdict, verdict)

    lines = [
        "## UNST33R PyRIT CI/CD Security Gate",
        "",
        f"**Verdict:** `{badge}`",
        "",
        "| Metric | Value | Threshold |",
        "| --- | --- | --- |",
        f"| Total probes | {summary['total_probes']} | - |",
        f"| Single-turn probes | {summary['single_turn_probes']} | - |",
        f"| Multi-turn probes | {summary['multi_turn_probes']} | - |",
        f"| Jailbreaks | {summary['jailbreak_count']} | - |",
        f"| Attack success rate | {_pct(summary['jailbreak_rate'])} "
        f"| <= {_pct(summary['max_jailbreak_rate'])} |",
        f"| Attacker refusals | {summary['attacker_refusals']} | - |",
        f"| Attacker refusal rate | {_pct(summary['attacker_refusal_rate'])} "
        f"| <= {_pct(summary['max_attacker_refusal_rate'])} |",
        f"| Benign probes | {summary['benign_probes']} | - |",
        f"| Benign false-refusal rate | "
        f"{_pct(summary['benign_false_refusal_rate'])} | info |",
        f"| Execution errors | {summary['execution_errors']} | - |",
        "",
        "### Per-category attack success rate",
        "",
        "| Category | Probes | Jailbreaks | ASR |",
        "| --- | --- | --- | --- |",
    ]
    for cat in sorted(summary["per_category"]):
        slot = summary["per_category"][cat]
        lines.append(
            f"| `{cat}` | {slot['probes']} | {slot['jailbreaks']} "
            f"| {_pct(slot['asr'])} |")

    lines += [
        "",
        "### Per-mode attack success rate",
        "",
        "| Mode | Probes | Jailbreaks | ASR |",
        "| --- | --- | --- | --- |",
    ]
    for mode in sorted(summary["per_mode"]):
        slot = summary["per_mode"][mode]
        lines.append(
            f"| {mode} | {slot['probes']} | {slot['jailbreaks']} "
            f"| {_pct(slot['asr'])} |")

    cost = summary["cost_estimate"]
    lines += [
        "",
        "### Estimated UNST33R cost",
        "",
        "| Model | Role tokens (in/out) | Estimated USD |",
        "| --- | --- | --- |",
    ]
    for key in sorted(cost["per_model"]):
        m = cost["per_model"][key]
        lines.append(
            f"| `{m['model']}` | {m['prompt_tokens']} / "
            f"{m['completion_tokens']} | ${m['estimated_usd']:.4f} |")
    lines.append(
        f"| **Total** | - | **${cost['total_estimated_usd']:.4f}** |")

    lines += [
        "",
        "### Run details",
        "",
        f"- Engine: `{summary['engine']}`",
        f"- Attacker (single-turn): `{summary['models']['attacker_single_turn']}`",
        f"- Attacker (multi-turn): `{summary['models']['attacker_multi_turn']}`",
        f"- Judge / scorer: `{summary['models']['scorer']}`",
        f"- Target: `{summary['models']['target']}`",
        f"- Corpus SHA-256: `{summary['corpus_sha256']}`",
        f"- Elapsed: {summary['elapsed_s']}s",
        "",
    ]
    return "\n".join(lines)
