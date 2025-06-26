import json
import os
from datetime import datetime

PROFILE_DIR = os.path.join(os.path.dirname(__file__), 'user_profiles')
os.makedirs(PROFILE_DIR, exist_ok=True)

def get_profile_path(user_id):
    return os.path.join(PROFILE_DIR, f'{user_id}.json')

def load_user_profile(user_id):
    path = get_profile_path(user_id)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Default profile
    return {
        "user_id": user_id,
        "preferences": {
            "response_style": "short",
            "tts_voice": "en-GB-Sonia",
            "topics": []
        },
        "interaction_history": []
    }

def save_user_profile(profile):
    path = get_profile_path(profile['user_id'])
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)

def update_preferences(user_id, new_prefs):
    profile = load_user_profile(user_id)
    profile['preferences'].update(new_prefs)
    save_user_profile(profile)

# Example: update_preferences('1234', {"response_style": "long"})
