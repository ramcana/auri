"""
Tests for IntentClassifier — heuristic routing signal.

Covers:
- Attachment-based signals (vision, document)
- Coding-pattern detection
- Web-search pattern detection
- Default chat fallback
- Signal field populated on match
"""
import pytest

from auri.intent import Intent, IntentClassifier


@pytest.fixture()
def clf():
    return IntentClassifier()


# ── Attachment priority ────────────────────────────────────────────────────────

def test_image_attachment_overrides_text(clf):
    intent = clf.classify("debug this function", has_image=True)
    assert intent.task == "vision"
    assert "image_attachment" in intent.signals


def test_file_attachment_overrides_text(clf):
    intent = clf.classify("summarize this report", has_file=True)
    assert intent.task == "document"
    assert "file_attachment" in intent.signals


def test_image_beats_file(clf):
    # image takes priority over file attachment
    intent = clf.classify("explain this diagram", has_image=True, has_file=True)
    assert intent.task == "vision"


# ── Coding patterns ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "```python\nprint('hello')\n```",
    "def my_function(x, y):",
    "import os",
    "fix the bug in this code",
    "debug the segfault",
    "refactor this into smaller functions",
    "syntax error on line 3",
    "type error: expected str",
    "traceback most recent call last",
    "help me with python",
    "write a javascript function",
    "git commit -m 'add feature'",
    "npm install express",
])
def test_coding_patterns(clf, text):
    intent = clf.classify(text)
    assert intent.task == "coding", f"Expected coding for: {text!r}"
    assert any("coding:" in s for s in intent.signals)


# ── Web patterns ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "search for latest nodejs releases",
    "what is the current version of nodejs",
    "latest news today",
    "what is the stock price of apple",
    "what's the weather today",
    "who is the CEO of OpenAI",
])
def test_web_patterns(clf, text):
    intent = clf.classify(text)
    assert intent.task == "web", f"Expected web for: {text!r}"
    assert any("web:" in s for s in intent.signals)


# ── Default fallback ───────────────────────────────────────────────────────────

def test_generic_text_falls_back_to_chat(clf):
    intent = clf.classify("Tell me about the French Revolution")
    assert intent.task == "chat"
    assert intent.signals == ["default"]


def test_empty_string_falls_back_to_chat(clf):
    intent = clf.classify("")
    assert intent.task == "chat"


def test_no_attachments_no_keywords_is_chat(clf):
    intent = clf.classify("How are you doing today?", has_image=False, has_file=False)
    assert intent.task == "chat"
