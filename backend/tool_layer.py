import requests
import re
from datetime import datetime
from backend.function_router import FunctionRouter

# --- Knowledge Maps ---
LOCATION_MAP = {
    "nyc": "America/New_York",
    "new york": "America/New_York",
    "la": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "tokyo": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "niagara falls": "America/Toronto",
}

SPECIFIC_ENTITY_MAP = {
    "president of the united states": "President_of_the_United_States",
    "president of usa": "President_of_the_United_States",
    "president of the us": "President_of_the_United_States",
}

# --- Tool Implementations ---

def get_current_time(location: str) -> str:
    """Fetches time for a location, using a map and falling back to a search."""
    search_location = location.lower()
    found_tz = LOCATION_MAP.get(search_location)
    if not found_tz:
        try:
            timezones_url = "http://worldtimeapi.org/api/timezone"
            tz_resp = requests.get(timezones_url, timeout=5)
            tz_resp.raise_for_status()
            timezones = tz_resp.json()
            search_term = search_location.replace(' ', '_')
            for tz in timezones:
                if f"/{search_term}" == tz.lower().split('/')[-1]:
                    found_tz = tz
                    break
            if not found_tz:
                for tz in timezones:
                    if search_term in tz.lower():
                        found_tz = tz
                        break
        except requests.RequestException:
            return ""
    if not found_tz:
        return ""
    try:
        time_url = f"http://worldtimeapi.org/api/timezone/{found_tz}"
        time_resp = requests.get(time_url, timeout=5)
        time_resp.raise_for_status()
        data = time_resp.json()
        datetime_str = data.get('datetime') or data.get('utc_datetime')
        if datetime_str:
            dt_obj = datetime.fromisoformat(datetime_str)
            return f"The current time in {location.title()} is {dt_obj.strftime('%-I:%M %p')} ({found_tz.replace('_', ' ')})."
    except requests.RequestException:
        return ""
    return ""

def get_wikipedia_summary(entity: str) -> str:
    """Fetch a summary for an entity from Wikipedia."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get('extract', '')
    except (requests.RequestException, ValueError):
        return ""

# --- Tool Router ---

router_instance = FunctionRouter()

def get_real_world_context(user_message: str) -> str:
    """Acts as a router to check for tool triggers and returns combined context."""
    lower_msg = user_message.lower()
    contexts = []

    # 1. Time Tool (legacy)
    time_match = re.search(r"time in (.*?)(?:\?|$)", lower_msg)
    if time_match:
        location = time_match.group(1).strip()
        if location:
            time_info = get_current_time(location)
            if time_info:
                contexts.append(time_info)

    # 2. Wikipedia / Entity Lookup Tool (legacy)
    entity_triggers = ["who is", "who's", "whos", "who was", "tell me about", "what is", "what's"]
    is_entity_query = any(trigger in lower_msg for trigger in entity_triggers) and "time in" not in lower_msg

    if is_entity_query:
        entity_query = ""
        for trigger in entity_triggers:
            match = re.search(r'\b' + trigger + r'\b\s*(.*)', lower_msg)
            if match:
                entity_query = match.group(1).strip().replace('?', '')
                break
        
        if entity_query:
            summary = ""
            # Check if the extracted query CONTAINS a known, specific entity phrase.
            for phrase, page_title in SPECIFIC_ENTITY_MAP.items():
                if phrase in entity_query:
                    summary = get_wikipedia_summary(page_title)
                    break # Found a specific match, stop searching
            
            # If no specific match was found, try a direct lookup with the extracted query.
            if not summary:
                summary = get_wikipedia_summary(entity_query.title())

            if summary:
                contexts.append(summary)

    # 3. Modular Tool Routing (new)
    module_name, func_name, result = router_instance.route(user_message)
    if result:
        contexts.append(str(result))

    # 4. Weather Tool (legacy placeholder)
    if any(word in lower_msg for word in ["weather", "sunny", "rainy", "temperature"]):
        contexts.append("It’s 22°C and sunny.")

    # 5. News Tool (legacy placeholder, now handled by module)
    # (Removed: handled by modules/news_card)

    if contexts:
        return f"Here’s some real-world context: {' '.join(contexts)}"

    return ""
