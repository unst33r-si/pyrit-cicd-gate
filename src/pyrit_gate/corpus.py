"""Deterministic adversarial-corpus loading and expansion.

``seeds/objectives.yaml`` contains 40 base objectives across 4 attack
categories plus 25 mutation framing strategies.  The Cartesian product
(40 x 25 = 1,000 unique adversarial probes) is expanded deterministically:

* ordering is derived from SHA-256 over the corpus file, so the same seeds
  always produce the same probes in the same order (reproducible digests);
* ``single_turn_count`` probes become single-turn prompts (the mutation is
  pre-baked into the prompt text);
* ``multi_turn_count`` probes are selected from the highest-risk categories
  for multi-turn ``RedTeamingAttack`` escalation;
* ``benign_security_lookalikes`` are legitimate defensive-security prompts
  used to measure the target's false-refusal rate.  They are *not* counted
  in ``total_probes`` / ASR.

PyYAML is used when installed; otherwise a small built-in parser handles the
restricted YAML subset used by the seeds file (nested maps, scalar lists,
``- key: value`` records, 2-space indentation).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

RISK_RANK = {"medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class Probe:
    """One adversarial probe to execute."""

    probe_id: str
    mode: str  # "single_turn" | "multi_turn" | "benign"
    category: str
    objective: str
    prompt: str
    mutation: str
    risk_rank: int = 0
    sort_key: str = ""


@dataclass
class Corpus:
    """Expanded probe corpus plus integrity metadata."""

    single_turn: List[Probe]
    multi_turn: List[Probe]
    benign: List[Probe]
    sha256: str
    seeds_path: str
    version: int = 1

    @property
    def probes(self) -> List[Probe]:
        return self.single_turn + self.multi_turn

    @property
    def total_probes(self) -> int:
        return len(self.single_turn) + len(self.multi_turn)


# ---------------------------------------------------------------------------
# YAML loading (PyYAML or built-in fallback)
# ---------------------------------------------------------------------------

def _unquote_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        inner = value[1:-1]
        return inner.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("''", "'")
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def mini_yaml_load(text: str) -> Any:
    """Parse the restricted YAML subset used by ``seeds/objectives.yaml``.

    Supported: nested mappings, ``key: scalar``, ``key:`` followed by an
    indented block, ``- scalar`` list items, ``- key: value`` record items
    with deeper-indented continuation keys, and full-line ``#`` comments.
    Double/single-quoted scalars and inline ``\\n`` escapes are handled.
    """
    rows = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        rows.append((indent, raw.strip()))
    if not rows:
        return None
    pos = [0]

    def parse_dict(indent: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        while pos[0] < len(rows):
            ind, content = rows[pos[0]]
            if ind < indent or content.startswith("- ") or content == "-":
                break
            if ind > indent:
                raise ValueError(f"mini-yaml: unexpected indent before {content!r}")
            key, sep, val = content.partition(":")
            if not sep:
                raise ValueError(f"mini-yaml: expected 'key:' line, got {content!r}")
            key = key.strip()
            val = val.strip()
            pos[0] += 1
            if val == "":
                if pos[0] < len(rows) and rows[pos[0]][0] > indent:
                    out[key] = parse_block(rows[pos[0]][0])
                else:
                    out[key] = None
            else:
                out[key] = _unquote_scalar(val)
        return out

    def parse_list(indent: int) -> List[Any]:
        out: List[Any] = []
        while pos[0] < len(rows):
            ind, content = rows[pos[0]]
            if ind < indent or not (content.startswith("- ") or content == "-"):
                break
            if ind > indent:
                raise ValueError(f"mini-yaml: unexpected indent before {content!r}")
            item = content[1:].strip()
            pos[0] += 1
            if (":" in item and not item.startswith(('"', "'"))
                    and item.split(":", 1)[0].strip().isidentifier()):
                key, _, val = item.partition(":")
                record: Dict[str, Any] = {}
                val = val.strip()
                record[key.strip()] = _unquote_scalar(val) if val else None
                while (pos[0] < len(rows) and rows[pos[0]][0] > indent
                       and not rows[pos[0]][1].startswith("- ")):
                    record.update(parse_dict(rows[pos[0]][0]))
                out.append(record)
            else:
                out.append(_unquote_scalar(item))
        return out

    def parse_block(indent: int) -> Any:
        content = rows[pos[0]][1]
        if content.startswith("- ") or content == "-":
            return parse_list(indent)
        return parse_dict(indent)

    return parse_block(rows[0][0])


def _load_seed_document(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError:
        doc = mini_yaml_load(text)
        if not isinstance(doc, dict):
            raise ValueError(f"failed to parse seeds file without PyYAML: {path}")
        return doc
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"seeds file did not produce a mapping: {path}")
    return loaded


# ---------------------------------------------------------------------------
# Deterministic expansion
# ---------------------------------------------------------------------------

def corpus_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _objective_variants(objective: str) -> Dict[str, str]:
    words = objective.split()
    frags = " + ".join(f"'{w}'" for w in words) if words else "''"
    half = max(1, len(objective) // 2)
    return {
        "objective": objective,
        "objective_lc": (objective[0].lower() + objective[1:]) if objective else objective,
        "objective_b64": base64.b64encode(objective.encode("utf-8")).decode("ascii"),
        "objective_hex": objective.encode("utf-8").hex(),
        "objective_frags": frags,
        "objective_half1": objective[:half].rstrip(),
        "objective_half2": objective[half:].lstrip(),
    }


def render_template(template: str, objective: str) -> str:
    """Render a mutation template against an objective.

    Supported placeholders: ``{objective}``, ``{objective_lc}``,
    ``{objective_b64}``, ``{objective_hex}``, ``{objective_frags}``,
    ``{objective_half1}`` and ``{objective_half2}``.

    Plain string replacement is used (not ``str.format``) so templates may
    contain literal JSON braces.
    """
    out = template
    for key, value in _objective_variants(objective).items():
        out = out.replace("{" + key + "}", value)
    return out


def _pair_sort_key(digest: str, category: str, obj_index: int, mut_id: str) -> str:
    return hashlib.sha256(
        f"{digest}:{category}:{obj_index}:{mut_id}".encode("utf-8")
    ).hexdigest()


def load_corpus(
    seeds_path: str,
    single_turn_count: int = 900,
    multi_turn_count: int = 100,
) -> Corpus:
    """Load seeds and deterministically expand into probe lists."""
    doc = _load_seed_document(seeds_path)
    digest = corpus_digest(seeds_path)

    categories = doc.get("categories") or {}
    strategies = doc.get("mutation_strategies") or []
    if not categories:
        raise ValueError(f"seeds file has no categories: {seeds_path}")
    if not strategies:
        raise ValueError(f"seeds file has no mutation_strategies: {seeds_path}")

    pairs = []
    for category, block in categories.items():
        objectives = (block or {}).get("objectives") or []
        rank = RISK_RANK.get(str((block or {}).get("risk", "high")).lower(), 2)
        for oi, objective in enumerate(objectives):
            for mut in strategies:
                mut_id = str(mut.get("id"))
                template = str(mut.get("template", "{objective}"))
                pairs.append({
                    "category": category,
                    "objective": str(objective),
                    "prompt": render_template(template, str(objective)),
                    "mutation": mut_id,
                    "risk_rank": rank,
                    "pair_key": f"{category}:{oi}:{mut_id}",
                    "sort_key": _pair_sort_key(digest, category, oi, mut_id),
                })

    # Deterministic global ordering.
    pairs.sort(key=lambda p: p["sort_key"])

    # Multi-turn pool: highest-risk categories first, stable order inside.
    mt_pool = sorted(pairs, key=lambda p: (-p["risk_rank"], p["sort_key"]))
    mt_pairs = mt_pool[:multi_turn_count]
    mt_keys = {p["pair_key"] for p in mt_pairs}

    st_pairs = [p for p in pairs if p["pair_key"] not in mt_keys]
    st_pairs = st_pairs[:single_turn_count]

    single_turn = [
        Probe(
            probe_id=f"st-{i + 1:04d}",
            mode="single_turn",
            category=p["category"],
            objective=p["objective"],
            prompt=p["prompt"],
            mutation=p["mutation"],
            risk_rank=p["risk_rank"],
            sort_key=p["sort_key"],
        )
        for i, p in enumerate(st_pairs)
    ]
    multi_turn = [
        Probe(
            probe_id=f"mt-{i + 1:04d}",
            mode="multi_turn",
            category=p["category"],
            objective=p["objective"],
            prompt=p["prompt"],
            mutation=p["mutation"],
            risk_rank=p["risk_rank"],
            sort_key=p["sort_key"],
        )
        for i, p in enumerate(mt_pairs)
    ]
    benign = [
        Probe(
            probe_id=f"bn-{i + 1:02d}",
            mode="benign",
            category="benign_security_lookalikes",
            objective=str(text),
            prompt=str(text),
            mutation="direct",
            risk_rank=0,
            sort_key=hashlib.sha256(f"{digest}:benign:{i}".encode()).hexdigest(),
        )
        for i, text in enumerate(doc.get("benign_security_lookalikes") or [])
    ]
    return Corpus(
        single_turn=single_turn,
        multi_turn=multi_turn,
        benign=benign,
        sha256=digest,
        seeds_path=os.path.abspath(seeds_path),
        version=int(doc.get("version") or 1),
    )
