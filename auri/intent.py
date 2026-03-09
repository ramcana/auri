"""
Heuristic intent classifier.

Classifies the user's message into a task type using regex patterns.
No LLM required — fast, offline, zero latency.

Task types:
  coding   — user is asking about or wants help with code
  vision   — an image is attached
  document — a non-image file is attached
  web      — user wants current/live information
  chat     — default general conversation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Intent:
    task: str                        # "coding" | "vision" | "document" | "web" | "chat"
    signals: list[str] = field(default_factory=list)  # which patterns fired (for transparency panel)


class IntentClassifier:
    # Patterns that indicate a coding-related message
    _CODING: list[str] = [
        r"```",                                         # fenced code block
        r"\bdef [a-z_]\w*\s*\(",                        # Python function definition
        r"\bclass [A-Z]\w*",                            # class definition
        r"\bimport \w+",                                # import statement
        r"\bfix\b.{0,40}\bbug\b",                       # "fix the bug"
        r"\bdebug\b",
        r"\brefactor\b",
        r"\bsyntax error\b",
        r"\btype error\b",
        r"\btraceback\b",
        r"\bpython\b|\bjavascript\b|\btypescript\b|\brust\b|\bgolang\b|\bjava\b|\bc\+\+\b",
        r"\bfunction\b|\bmethod\b|\bmodule\b|\bpackage\b|\blibrary\b",
        r"\bgit \w+",                                   # git commands
        r"\bnpm \w+|\bpip \w+|\bcargo \w+",             # package managers
    ]

    # Patterns that indicate a request for live/current information
    _WEB: list[str] = [
        r"\bsearch\b.{0,30}\bfor\b",
        r"\blatest\b|\bcurrent\b.{0,20}\b(news|price|version|release)\b",
        r"\bwhat.{0,10}happening\b",
        r"\bstock price\b|\bweather\b|\btoday.{0,10}news\b",
        # "who is" only fires for web-specific targets; generic "who is the author"
        # should not be classified as web — it may refer to an ingested document
        r"\bwho is\b.{0,30}\b(ceo|cto|president|founder|director)\b",
        r"\bwhat is\b.{0,30}\b(company|startup|person|ceo)\b",
        r"\bwikipedia\b",
    ]

    def classify(
        self,
        text: str,
        has_image: bool = False,
        has_file: bool = False,
    ) -> Intent:
        """Return an Intent for the given message and attachment state."""
        if has_image:
            return Intent(task="vision", signals=["image_attachment"])

        if has_file:
            return Intent(task="document", signals=["file_attachment"])

        lower = text.lower()

        for pattern in self._CODING:
            if re.search(pattern, lower):
                return Intent(task="coding", signals=[f"coding:{pattern}"])

        for pattern in self._WEB:
            if re.search(pattern, lower):
                return Intent(task="web", signals=[f"web:{pattern}"])

        return Intent(task="chat", signals=["default"])
