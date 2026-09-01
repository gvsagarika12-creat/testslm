"""Slice the 3H spec so each call carries only the sections it needs.

The full spec is ~5,800 tokens and was being sent on every request, including
MODE 1-5 rules that a simulation turn never consults. On a 12B model that is
paid twice — once in prompt processing latency, once in the attention the model
spends on irrelevant instructions.

Nothing is edited or deleted: prompts/3h_agent.md remains the whole contract,
and this module quotes from it. A section that is not listed for a mode is
simply not sent.
"""
from __future__ import annotations

import re
from pathlib import Path

from ragforge.config import Settings, settings as default_settings

AGENT_FILE = "3h_agent.md"
DEPLOYMENT_FILE = "3h_deployment.md"

# §0 (absolute constraints), §1 (identity) and §2 (the framework) are the
# agent's identity and are sent for every mode. The rest is per-mode.
_ALWAYS = ("0", "1", "2")

MODE_SECTIONS: dict[str, tuple[str, ...]] = {
    # Teaching needs alignment, the load governor, and the style budgets.
    "teach": _ALWAYS + ("3", "5", "9", "11"),
    # Grading needs the rubrics and the escalation rules, not the load governor.
    "assess": _ALWAYS + ("8", "9", "10"),
    # Simulation needs the rubrics (for the debrief) and escalation.
    "simulate": _ALWAYS + ("8", "9", "10", "11"),
}

MODE_BLOCKS: dict[str, tuple[str, ...]] = {
    "teach": ("MODE 3",),
    "assess": ("MODE 4", "MODE 5"),
    "simulate": ("MODE 4", "MODE 5", "MODE 6"),
}

# Which D-sections of the deployment addendum apply to each mode.
DEPLOYMENT_BLOCKS: dict[str, tuple[str, ...]] = {
    "teach": ("D1", "D2", "D3", "D3a", "D4", "D5"),
    "assess": ("D1", "D2", "D4", "D6"),
    "simulate": ("D1", "D2", "D4", "D7"),
}

_SECTION = re.compile(r"^## (\d+)\.\s", re.MULTILINE)
_MODE = re.compile(r"^### .*?(MODE \d)", re.MULTILINE)
_DEPLOY = re.compile(r"^## (D\d+[a-z]?)\s*[—-]", re.MULTILINE)


def _split(text: str, pattern: re.Pattern) -> dict[str, str]:
    """Split markdown into {key: block} at each heading the pattern matches."""
    matches = list(pattern.finditer(text))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.start() : end].rstrip()
    return blocks


def load_raw(config: Settings | None = None) -> tuple[str, str]:
    """The two prompt files, verbatim."""
    config = config or default_settings
    directory = Path(config.prompts_dir)
    contents = []
    for name in (AGENT_FILE, DEPLOYMENT_FILE):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing prompt file: {path}")
        contents.append(path.read_text(encoding="utf-8"))
    return contents[0], contents[1]


def build_system_prompt(mode: str, config: Settings | None = None) -> str:
    """The spec sections relevant to `mode`, plus the deployment addendum.

    An unknown mode returns the whole spec — the safe direction to fail, since
    an over-long prompt is slow while a missing rule is a contract violation.
    """
    agent, deployment = load_raw(config)

    if mode not in MODE_SECTIONS:
        return f"{agent}\n\n---\n\n{deployment}"

    sections = _split(agent, _SECTION)
    modes = _split(sections.get("7", ""), _MODE)
    deploy_blocks = _split(deployment, _DEPLOY)

    # The document title and deployment-target line sit above section 0. They
    # are the framing every mode is read under, so they are never sliced out.
    first = _SECTION.search(agent)
    preamble = agent[: first.start()].strip() if first else ""

    parts = [preamble] if preamble else []
    parts.extend(sections[key] for key in MODE_SECTIONS[mode] if key in sections)
    parts.extend(modes[key] for key in MODE_BLOCKS[mode] if key in modes)
    parts.append("---\n\n# RAGForge deployment addendum")
    parts.extend(
        deploy_blocks[key] for key in DEPLOYMENT_BLOCKS[mode] if key in deploy_blocks
    )
    return "\n\n".join(parts)
