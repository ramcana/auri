"""
ContextPacker: fits all context inputs within a model's token limit.

Packing priority (highest to lowest):
  1. System prompt    — always included in full, never truncated
  2. Latest user turn — always included (it is the current query)
  3. Chat history     — sliding window; oldest turns dropped first

Token estimation: 1 token ≈ 4 characters (fast heuristic, no tokenizer needed).
Exact tokenization is model-specific and adds latency — the heuristic is good enough
for routing and budget decisions.

Future: tool outputs and retrieved docs are additional inputs. The priority order
when those arrive will be:
  system > latest user message > tool outputs > retrieved docs > older history
"""

from __future__ import annotations

from dataclasses import dataclass

_CHARS_PER_TOKEN = 4  # conservative heuristic (4 chars ≈ 1 token)
_MSG_OVERHEAD = 4     # per-message overhead tokens (role, formatting)


@dataclass
class PackedContext:
    messages: list[dict]          # ready to pass to the inference API
    history_turns_included: int   # how many history turns fit
    history_turns_total: int      # how many there were before packing
    estimated_tokens: int         # rough prompt token estimate
    truncated: bool               # True if any history was dropped


class ContextPacker:
    """Pack chat history into a prompt that fits within the model's context limit."""

    def __init__(self, chars_per_token: int = _CHARS_PER_TOKEN) -> None:
        self._cpt = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self._cpt)

    def _msg_tokens(self, msg: dict) -> int:
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Multimodal content blocks
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        return self.estimate_tokens(str(content)) + _MSG_OVERHEAD

    def pack(
        self,
        system_prompt: str,
        history: list[dict],
        context_limit: int,
        output_budget: int = 2048,
    ) -> PackedContext:
        """Pack history into at most (context_limit - output_budget) tokens.

        Args:
            system_prompt: the full system prompt text.
            history: chat messages in chronological order, WITHOUT the system message.
                     The last message is assumed to be the current user turn and is
                     always included.
            context_limit: model's max_model_len (total token capacity).
            output_budget: tokens reserved for the model's response.

        Returns:
            PackedContext with the message list ready for the API.
        """
        available = max(256, context_limit - output_budget)

        system_msg = {"role": "system", "content": system_prompt}
        used = self._msg_tokens(system_msg)

        if not history:
            return PackedContext(
                messages=[system_msg],
                history_turns_included=0,
                history_turns_total=0,
                estimated_tokens=used,
                truncated=False,
            )

        # Walk backwards: newest messages have highest priority.
        # The last message (current user turn) is always included first.
        included: list[dict] = []
        for msg in reversed(history):
            cost = self._msg_tokens(msg)
            if used + cost <= available:
                included.append(msg)
                used += cost
            else:
                # Could try to include later turns that are smaller, but FIFO
                # dropping is simpler and avoids out-of-order context.
                break

        included.reverse()  # restore chronological order

        truncated = len(included) < len(history)

        return PackedContext(
            messages=[system_msg] + included,
            history_turns_included=len(included),
            history_turns_total=len(history),
            estimated_tokens=used,
            truncated=truncated,
        )
