import requests
import re
import os # For environment variables
import logging # For logging API calls
from datetime import datetime
from backend.function_router import FunctionRouter

logger = logging.getLogger(__name__)

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

def get_weather_forecast(location_query: str) -> str:
    """Fetches weather forecast for a location using Open-Meteo."""
    # Attempt to get coordinates for the location_query using Nominatim (courtesy of Open-Meteo)
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        geo_params = {'name': location_query, 'count': 1, 'language': 'en', 'format': 'json'}
        geo_resp = requests.get(geocoding_url, params=geo_params, timeout=5)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get('results'):
            logger.warning(f"No geocoding results for location: {location_query}")
            return f"Sorry, I couldn't find the location {location_query} for weather information."

        location_info = geo_data['results'][0]
        latitude = location_info.get('latitude')
        longitude = location_info.get('longitude')
        resolved_location_name = location_info.get('name', location_query)
        admin1 = location_info.get('admin1', '') # State or region
        country = location_info.get('country_code', '')
        display_name_parts = [resolved_location_name]
        if admin1: display_name_parts.append(admin1)
        if country: display_name_parts.append(country)
        resolved_display_name = ", ".join(filter(None,display_name_parts))


        if latitude is None or longitude is None:
            logger.warning(f"Could not get coordinates for {location_query}")
            return f"Sorry, I couldn't get coordinates for {location_query}."

        # Get weather forecast using coordinates
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current_weather': 'true', # Get current weather
            'temperature_unit': 'celsius',
            'windspeed_unit': 'kmh',
            'precipitation_unit': 'mm',
            'forecast_days': 1 # Just current day
        }
        weather_resp = requests.get(weather_url, params=params, timeout=5)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        current_weather = weather_data.get('current_weather')
        if current_weather:
            temp = current_weather.get('temperature')
            windspeed = current_weather.get('windspeed')
            weather_code = current_weather.get('weathercode') # WMO Weather interpretation codes

            # Basic weather code interpretation (can be expanded)
            # Ref: https://open-meteo.com/en/docs#weathervariables
            weather_desc = "clear sky"
            if weather_code is not None:
                if weather_code == 0: weather_desc = "clear sky"
                elif weather_code == 1: weather_desc = "mainly clear"
                elif weather_code == 2: weather_desc = "partly cloudy"
                elif weather_code == 3: weather_desc = "overcast"
                elif weather_code in [45, 48]: weather_desc = "fog"
                elif weather_code in [51, 53, 55, 56, 57]: weather_desc = "drizzle"
                elif weather_code in [61, 63, 65, 66, 67]: weather_desc = "rain"
                elif weather_code in [71, 73, 75, 77]: weather_desc = "snowfall"
                elif weather_code in [80, 81, 82]: weather_desc = "rain showers"
                elif weather_code in [85, 86]: weather_desc = "snow showers"
                elif weather_code in [95, 96, 99]: weather_desc = "thunderstorm"
                else: weather_desc = "unknown conditions"

            return (f"The current weather in {resolved_display_name} is: "
                    f"{weather_desc}, with a temperature of {temp}°C "
                    f"and wind speed of {windspeed} km/h.")
        else:
            return f"Sorry, I couldn't retrieve current weather for {resolved_display_name}."

    except requests.RequestException as e:
        logger.error(f"Weather API request failed for {location_query}: {e}")
        return f"Sorry, I couldn't fetch the weather information due to a network error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_weather_forecast for {location_query}: {e}", exc_info=True)
        return "Sorry, an unexpected error occurred while fetching weather information."


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
        # Ensure result is a string and not None before appending
        contexts.append(str(result))

    # 4. Weather Tool (Improved)
    # Example trigger: "what's the weather in London?" or "weather in Paris"
    # More robust trigger considering "weather in/for <location>"
    weather_match = re.search(r"weather (?:in|for) (.*?)(?:\?|$)", lower_msg)
    if weather_match:
        location = weather_match.group(1).strip()
        if location:
            # Avoid calling weather if it's part of a time query, e.g. "what's the time and weather in London"
            # This is a simple heuristic; more advanced NLP could parse combined intents better.
            if not ("time in" in lower_msg and location in lower_msg.split("time in")[1]):
                weather_info = get_weather_forecast(location)
                if weather_info:
                    contexts.append(weather_info)
    elif any(word in lower_msg for word in ["weather", "forecast"]) and not weather_match:
        # If "weather" or "forecast" is mentioned but not in the "weather in <location>" pattern,
        # try to extract a location if it was mentioned earlier in the message for other tools.
        # This is a simplistic approach. For better context, conversation history would be needed here.
        # For now, we'll just indicate that a location is needed if no specific pattern matches.
        # Or, if a previous tool identified a location, that could be used.
        # This part is tricky without full NLP intent parsing.
        # For now, if only "weather" is asked without location, we might prompt back or use a default.
        # The current implementation of get_weather_forecast requires a location.
        # We could try to find a location from the LOCATION_MAP if mentioned.
        found_loc_for_general_weather = None
        for loc_keyword in LOCATION_MAP.keys():
            if loc_keyword in lower_msg:
                found_loc_for_general_weather = loc_keyword
                break
        if found_loc_for_general_weather:
            weather_info = get_weather_forecast(found_loc_for_general_weather)
            if weather_info:
                contexts.append(weather_info)
        # else:
            # contexts.append("Please specify a location for the weather forecast.") # Alternative

    # 5. News Tool (handled by modules/news_card)

    if contexts:
        return f"Here’s some real-world context: {' '.join(contexts)}"

    return ""
