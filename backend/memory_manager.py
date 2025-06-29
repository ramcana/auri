import os
import json
import logging
from datetime import datetime # For timestamps
from typing import Dict, List, Any, Optional # For type hinting
from .user_profile import load_user_profile

logger = logging.getLogger(__name__)

FACTS_DIR = os.path.join(os.path.dirname(__file__), 'user_facts')
os.makedirs(FACTS_DIR, exist_ok=True)

def get_facts_path(user_id: str) -> str:
    return os.path.join(FACTS_DIR, f'{user_id}_facts.json')

def _load_facts(user_id: str) -> List[Dict[str, Any]]:
    """Loads facts for a user from their JSON file."""
    facts_path = get_facts_path(user_id)
    if os.path.isfile(facts_path):
        try:
            with open(facts_path, 'r', encoding='utf-8') as f:
                facts = json.load(f)
                if isinstance(facts, list): # Ensure it's a list
                    return facts
                else: # If file content is not a list (e.g. old format or corrupted)
                    logger.warning(f"Facts file for user {user_id} is not a list. Initializing fresh facts.")
                    return []
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from facts file for user {user_id}. Initializing fresh facts.")
            return [] # Return empty list if file is corrupted
        except Exception as e:
            logger.error(f"Error loading facts for user {user_id}: {e}. Initializing fresh facts.")
            return []
    return []

def _save_facts(user_id: str, facts: List[Dict[str, Any]]):
    """Saves facts for a user to their JSON file."""
    facts_path = get_facts_path(user_id)
    try:
        with open(facts_path, 'w', encoding='utf-8') as f:
            json.dump(facts, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving facts for user {user_id}: {e}")


def add_user_fact(user_id: str, content: str, timestamp: Optional[str] = None):
    """Adds a structured user message to the conversation facts."""
    if not timestamp:
        timestamp = datetime.utcnow().isoformat()

    fact_entry = {
        "type": "user",
        "content": content,
        "timestamp": timestamp
    }

    facts = _load_facts(user_id)
    facts.append(fact_entry)
    _save_facts(user_id, facts)
    logger.debug(f"Added user fact for {user_id}: {content[:50]}...")

def add_assistant_fact(user_id: str, content: str,
                       tool_used: Optional[str] = None,
                       tool_input: Optional[Any] = None,
                       tool_output: Optional[Any] = None,
                       timestamp: Optional[str] = None):
    """Adds a structured assistant message (and optional tool usage) to the conversation facts."""
    if not timestamp:
        timestamp = datetime.utcnow().isoformat()

    fact_entry = {
        "type": "assistant",
        "content": content, # This would be the final textual response to the user
        "timestamp": timestamp
    }
    if tool_used:
        fact_entry["tool_used"] = tool_used
        fact_entry["tool_input"] = tool_input if tool_input is not None else "N/A"
        fact_entry["tool_output"] = tool_output if tool_output is not None else "N/A"

    facts = _load_facts(user_id)
    facts.append(fact_entry)
    _save_facts(user_id, facts)
    logger.debug(f"Added assistant fact for {user_id}: {content[:50]}...")


# Kept for compatibility or direct structured fact addition if needed, but prefer specific adders.
# Deprecated: Use add_user_fact or add_assistant_fact instead for clarity.
def add_fact(user_id: str, fact_entry: Dict[str, Any]):
    """
    Appends a new structured fact entry to the user's conversation facts.
    Ensures 'timestamp' and 'type' are present.
    DEPRECATED: Prefer add_user_fact or add_assistant_fact.
    """
    logger.warning("Direct use of add_fact is deprecated. Use add_user_fact or add_assistant_fact.")
    if "timestamp" not in fact_entry:
        fact_entry["timestamp"] = datetime.utcnow().isoformat()
    if "type" not in fact_entry:
        fact_entry["type"] = "generic" # Default if type not specified

    facts = _load_facts(user_id)
    facts.append(fact_entry)
    _save_facts(user_id, facts)


def get_conversation_facts(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve all structured facts for the user."""
    facts_path = get_facts_path(user_id)
    if os.path.isfile(facts_path):
        with open(facts_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def summarize_facts(user_id: str, max_turns: int = 5) -> str:
    """
    Return a string summarizing the most recent conversation turns for prompt injection.
    Each turn can consist of a user message and subsequent assistant response (and tool use).
    """
    facts = get_conversation_facts(user_id)
    if not facts:
        return ""

    # Take the last N facts (max_facts could translate to roughly max_turns * 2 if every turn has user+assistant)
    # A more sophisticated approach might count actual "turns".
    # For now, let's take `max_turns * 2` as an approximation, then filter/format.
    # Or, more simply, just take the last `max_turns` entries, assuming each is a distinct utterance.
    recent_facts = facts[-max_turns:]

    summary_parts = []
    for fact in recent_facts:
        fact_type = fact.get("type", "unknown")
        content = fact.get("content", "")
        timestamp = fact.get("timestamp", "") # Could be used if needed: {timestamp[:19]}

        if fact_type == "user":
            summary_parts.append(f"User: {content}")
        elif fact_type == "assistant":
            assistant_prefix = "Assistant:"
            tool_details = ""
            if fact.get("tool_used"):
                tool_used = fact.get("tool_used")
                tool_input = fact.get("tool_input", "N/A")
                # Tool output might be too verbose for direct history.
                # Consider summarizing it or just noting that a tool was used.
                # For now, let's just indicate tool usage.
                # assistant_prefix = f"Assistant (used tool {tool_used} with input '{tool_input}'):"
                assistant_prefix = f"Assistant (used {tool_used}):" # Simpler
            summary_parts.append(f"{assistant_prefix} {content}")
        else: # Generic or unknown, try to make it readable
            summary_parts.append(f"{str(fact_type).capitalize()}: {content if content else str(fact)}")


    if not summary_parts:
        return ""

    return "Recent conversation turns:\n" + "\n".join(summary_parts)

def get_user_context(user_id: str) -> str:
    """Return a summary string for LLM system prompt based on user profile."""
    profile = load_user_profile(user_id)
    prefs = profile.get('preferences', {})
    topics = ', '.join(prefs.get('topics', []))
    style = prefs.get('response_style', 'normal')
    return f"The user likes {style} answers and often asks about {topics if topics else 'various topics'}."
