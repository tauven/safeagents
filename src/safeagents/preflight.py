# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks a request before it reaches the model.

Secrets anywhere in the payload are masked with gitleaks' rules, so they never
leave for the model provider. The newest input, the part the agent hasn't seen
yet, is then put to the judge, which answers whether it is a prompt injection.
Tool calls are checked on the way back, in `guard.py`.
"""

from __future__ import annotations

import logging
import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from math import log2
from pathlib import Path

from .constants import DLP_ALLOWED, DLP_ALLOWED_PATTERNS, INJECTION_SYSTEM, MIN_INJECTION_LENGTH
from .judge import Judge, JudgeError
from .utils import content_text

logger = logging.getLogger("safeagents")

# gitleaks' rule file, from https://github.com/gitleaks/gitleaks (MIT).
DEFAULT_RULES_PATH = Path(__file__).with_name("gitleaks.toml")

# gitleaks patterns are written for Go's RE2, which spells two things differently.
POSIX_CLASSES = {"[[:alnum:]]": "[A-Za-z0-9]", "[[:alpha:]]": "[A-Za-z]", "[[:digit:]]": "[0-9]"}


def _compile(regex: str) -> re.Pattern[str]:
    """A gitleaks regex as a Python one: RE2 allows `(?i)` anywhere and `\\z`."""
    if "(?i)" in regex:
        regex = "(?i)" + regex.replace("(?i)", "")
    regex = regex.replace(r"\z", r"\Z")
    for go, python in POSIX_CLASSES.items():
        regex = regex.replace(go, python)
    return re.compile(regex)


def _entropy(text: str) -> float:
    """Shannon entropy per character, as gitleaks uses it to drop rule-shaped noise."""
    counts = Counter(text)
    return -sum(count / len(text) * log2(count / len(text)) for count in counts.values())


@dataclass(frozen=True)
class Allowlist:
    """Values matching these are not secrets: `true`, `${VAR}`, paths, test keys."""

    regexes: tuple[re.Pattern[str], ...] = ()
    stopwords: frozenset[str] = frozenset()

    def allows(self, secret: str) -> bool:
        return secret in self.stopwords or any(regex.search(secret) for regex in self.regexes)


def _allowlist(section: dict | list | None) -> Allowlist:
    """A gitleaks `allowlist` or `allowlists` section, as one `Allowlist`."""
    sections = section if isinstance(section, list) else [section or {}]
    regexes: list[re.Pattern[str]] = []
    stopwords: set[str] = set()
    for entry in sections:
        if not isinstance(entry, dict):
            continue
        for regex in entry.get("regexes") or ():
            try:
                regexes.append(_compile(regex))
            except re.error as e:
                logger.warning("[safeagents] dlp: skipping allowlist regex %r: %s", regex, e)
        stopwords.update(str(word) for word in entry.get("stopwords") or ())
    return Allowlist(tuple(regexes), frozenset(stopwords))


@dataclass(frozen=True)
class Rule:
    """One gitleaks rule: `group` is the capture holding the secret, if there is one."""

    name: str
    pattern: re.Pattern[str]
    keywords: tuple[str, ...]
    group: int
    entropy: float
    allow: tuple[Allowlist, ...]

    def wanted(self, secret: str) -> bool:
        if self.entropy and _entropy(secret) < self.entropy:
            return False
        if secret in DLP_ALLOWED or any(pattern.fullmatch(secret) for pattern in DLP_ALLOWED_PATTERNS):
            return False
        return not any(allowlist.allows(secret) for allowlist in self.allow)

    def replace(self, text: str) -> tuple[str, int]:
        """`text` with every secret this rule finds masked, and how many there were."""
        count = 0

        def substitute(match: re.Match[str]) -> str:
            nonlocal count
            start, end = match.span(self.group)
            if start < 0:
                return match.group(0)
            secret = match.group(0)[start - match.start() : end - match.start()]
            if not self.wanted(secret):
                return match.group(0)
            count += 1
            return (
                match.group(0)[: start - match.start()]
                + f"[REDACTED_{self.name}]"
                + match.group(0)[end - match.start() :]
            )

        return self.pattern.sub(substitute, text), count


class Rules:
    """gitleaks' ruleset, loaded once and applied with its keyword prefilter."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_RULES_PATH
        config = tomllib.loads(self.path.read_text(encoding="utf-8"))
        global_allow = _allowlist(config.get("allowlist"))
        self.rules: list[Rule] = []
        for rule in config.get("rules") or ():
            regex = rule.get("regex")
            if not regex:
                continue
            try:
                pattern = _compile(regex)
            except re.error as e:  # a rule Python can't run is skipped, not fatal
                logger.warning("[safeagents] dlp: skipping rule %s: %s", rule.get("id"), e)
                continue
            group = int(rule.get("secretGroup") or (1 if pattern.groups else 0))
            self.rules.append(
                Rule(
                    name=str(rule.get("id") or "secret").upper().replace("-", "_"),
                    pattern=pattern,
                    keywords=tuple(str(keyword).lower() for keyword in rule.get("keywords") or ()),
                    group=group if group <= pattern.groups else 0,
                    entropy=float(rule.get("entropy") or 0),
                    allow=(global_allow, _allowlist(rule.get("allowlist") or rule.get("allowlists"))),
                )
            )

    def redact(self, text: str) -> tuple[str, list[str]]:
        """`text` with secrets masked, and the names of the rules that fired."""
        if not text or not self.rules:
            return text, []
        lowered = text.lower()
        hits: list[str] = []
        for rule in self.rules:
            # gitleaks' own shortcut: a rule can only match if one of its keywords is there.
            if rule.keywords and not any(keyword in lowered for keyword in rule.keywords):
                continue
            text, count = rule.replace(text)
            hits.extend([rule.name] * count)
        return text, hits


def redact(value: object, rules: Rules, hits: list[str]) -> object:
    """`value` with every string in it redacted; `hits` collects what fired."""
    if isinstance(value, str):
        redacted, found = rules.redact(value)
        hits.extend(found)
        return redacted
    if isinstance(value, dict):
        return {key: redact(item, rules, hits) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, rules, hits) for item in value]
    return value


def newest_text(request: dict) -> str:
    """The messages since the last assistant turn: what this request adds to the conversation.

    That is the user's new input or the results of the last tool calls, the places
    injected instructions arrive from. Older turns were checked when they arrived.
    """
    parts: list[str] = []
    for message in reversed(request.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") == "assistant":
            break
        parts.append(_message_text(message.get("content")))
    return "\n\n".join(reversed([part for part in parts if part]))


def _message_text(content: object) -> str:
    """The text of one message, including Anthropic tool results."""
    if not isinstance(content, list):
        return content_text(content)
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            parts.append(content_text(block.get("content")))
        elif block.get("type") in ("text", "input_text") and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(part for part in parts if part)


@dataclass
class Preflight:
    """Masks secrets in a request and has the judge check its newest input.

    `rules=None` turns the masking off, `judge=None` the injection check.
    """

    rules: Rules | None = None
    judge: Judge | None = None
    # Block input the judge calls unsafe, and input it couldn't be asked about.
    block: bool = True
    fail_closed: bool = False

    async def check(self, request: dict) -> tuple[dict, str | None]:
        """The request to forward (the same object if nothing changed), or why it is refused."""
        if self.rules is not None:
            hits: list[str] = []
            redacted = redact(request, self.rules, hits)
            if hits:
                logger.warning("[safeagents] masked %s in the request", ", ".join(sorted(set(hits))))
                request = redacted
        refusal = await self._injection(request)
        return request, refusal if self.block else None

    async def _injection(self, request: dict) -> str | None:
        if self.judge is None:
            return None
        text = newest_text(request)
        if len(text) <= MIN_INJECTION_LENGTH:
            return None
        try:
            answer = await self.judge.ask(INJECTION_SYSTEM, f"<input>\n{text}\n</input>")
        except JudgeError as e:
            logger.warning("[safeagents] injection judge unavailable: %s", e)
            return "the injection judge is unavailable" if self.fail_closed else None
        if "UNSAFE" in answer.upper():
            return "the newest input looks like a prompt injection"
        return None
