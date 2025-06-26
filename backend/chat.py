from .user_profile import load_user_profile, save_user_profile, update_preferences
from .interaction_logger import log_interaction
from .memory_manager import get_user_context

# Example chat handler

def handle_message(user_id, message, tag=None, model=None):
    # Log interaction
    log_interaction(user_id, message, tag=tag, model=model)
    # Load/update user profile
    profile = load_user_profile(user_id)
    # Example: update topics if tag is new
    if tag and tag not in profile['preferences'].get('topics', []):
        profile['preferences']['topics'].append(tag)
        save_user_profile(profile)
    # Get user context for LLM
    user_context = get_user_context(user_id)
    # Inject user_context into LLM system prompt as needed
    return user_context

# Example usage:
# handle_message('1234', "What's the news?", tag="news", model="ollama")
