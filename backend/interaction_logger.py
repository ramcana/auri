import json
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), 'user_logs')
os.makedirs(LOG_DIR, exist_ok=True)

def get_log_path(user_id):
    return os.path.join(LOG_DIR, f'{user_id}_log.json')

def log_interaction(user_id, message, tag=None, model=None, tone=None, sentiment=None):
    log_path = get_log_path(user_id)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "tag": tag,
        "model": model,
        "tone": tone,
        "sentiment": sentiment
    }
    # Append to log file
    logs = []
    if os.path.isfile(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(entry)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)

# Example: log_interaction('1234', "What's the news?", tag="news")
