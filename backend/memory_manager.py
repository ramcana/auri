import os
import json
from .user_profile import load_user_profile

FACTS_DIR = os.path.join(os.path.dirname(__file__), 'user_facts')
os.makedirs(FACTS_DIR, exist_ok=True)

def get_facts_path(user_id):
    return os.path.join(FACTS_DIR, f'{user_id}_facts.json')

def add_fact(user_id, fact):
    """Append a new fact to the user's conversation facts."""
    facts_path = get_facts_path(user_id)
    facts = []
    if os.path.isfile(facts_path):
        with open(facts_path, 'r', encoding='utf-8') as f:
            try:
                facts = json.load(f)
            except Exception:
                facts = []
    facts.append(fact)
    with open(facts_path, 'w', encoding='utf-8') as f:
        json.dump(facts, f, indent=2)

def get_conversation_facts(user_id):
    """Retrieve all facts for the user."""
    facts_path = get_facts_path(user_id)
    if os.path.isfile(facts_path):
        with open(facts_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []

def summarize_facts(user_id, max_facts=5):
    """Return a summary string of the most recent conversation facts for prompt injection."""
    facts = get_conversation_facts(user_id)
    if not facts:
        return ""
    summary = '\n'.join(f"- {fact}" for fact in facts[-max_facts:])
    return f"Conversation facts so far:\n{summary}"

def get_user_context(user_id):
    """Return a summary string for LLM system prompt based on user profile."""
    profile = load_user_profile(user_id)
    prefs = profile.get('preferences', {})
    topics = ', '.join(prefs.get('topics', []))
    style = prefs.get('response_style', 'normal')
    return f"The user likes {style} answers and often asks about {topics if topics else 'various topics'}."
