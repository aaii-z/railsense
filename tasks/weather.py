"""
Weather info panel using Open-Meteo (free, no API key).

Provides current conditions and a short forecast for a given station/city.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

import requests

log = logging.getLogger(__name__)

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_WMO_ICONS = {
    0: "☀️",   1: "\U0001f324️", 2: "⛅",          3: "☁️",
    45: "\U0001f32b️", 48: "\U0001f32b️",
    51: "\U0001f326️", 53: "\U0001f326️", 55: "\U0001f327️",
    56: "\U0001f327️", 57: "\U0001f327️",
    61: "\U0001f326️", 63: "\U0001f327️", 65: "\U0001f327️",
    66: "\U0001f327️", 67: "\U0001f327️",
    71: "\U0001f328️", 73: "\U0001f328️", 75: "\U0001f328️", 77: "\U0001f328️",
    80: "\U0001f326️", 81: "\U0001f327️", 82: "\U0001f327️",
    85: "\U0001f328️", 86: "\U0001f328️",
    95: "⚡",         96: "⚡",          99: "⚡",
}


@dataclass
class WeatherInfo:
    location: str
    temperature_c: float
    feels_like_c: float
    condition: str
    icon: str
    wind_speed_kmh: float
    humidity_percent: int
    forecast_hours: list[dict]
    arrival_time: str | None = None


@lru_cache(maxsize=256)
def _geocode(place: str) -> tuple[float, float, str] | None:
    """Resolve a place name to (lat, lon, display_name). Cached."""
    try:
        resp = requests.get(
            _GEOCODE_URL,
            params={"name": place, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        r = results[0]
        return r["latitude"], r["longitude"], r.get("name", place)
    except Exception:
        log.debug("Geocoding failed for %r", place, exc_info=True)
        return None


def get_weather(place: str, arrival_time: str | None = None) -> WeatherInfo | None:
    """Fetch weather for a place, optionally at a specific arrival time.

    If arrival_time is given (ISO 8601 or HH:MM on a known date), returns
    the forecast for that hour instead of current conditions.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    geo = _geocode(place)
    if not geo and " " in place:
        geo = _geocode(place.split()[0])
    if not geo:
        return None
    lat, lon, name = geo

    try:
        resp = requests.get(
            _WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
                "hourly": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m,precipitation_probability",
                "forecast_days": 7,
                "timezone": "Europe/London",
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        log.debug("Weather fetch failed for %s", name, exc_info=True)
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    feels = hourly.get("apparent_temperature", [])
    codes = hourly.get("weather_code", [])
    winds = hourly.get("wind_speed_10m", [])
    humids = hourly.get("relative_humidity_2m", [])
    precip = hourly.get("precipitation_probability", [])

    # Find the hourly index closest to the arrival time
    arrival_idx = None
    arrival_label = None
    if arrival_time:
        try:
            uk_tz = ZoneInfo("Europe/London")
            if "T" in str(arrival_time):
                arr_dt = datetime.fromisoformat(str(arrival_time).replace("Z", "+00:00"))
                if arr_dt.tzinfo is None:
                    arr_dt = arr_dt.replace(tzinfo=uk_tz)
            else:
                arr_dt = None

            if arr_dt and times:
                arr_str = arr_dt.strftime("%Y-%m-%dT%H:00")
                if arr_str in times:
                    arrival_idx = times.index(arr_str)
                else:
                    # Find closest hour
                    for i, t in enumerate(times):
                        t_dt = datetime.fromisoformat(t).replace(tzinfo=uk_tz)
                        if t_dt >= arr_dt:
                            arrival_idx = i
                            break
                if arrival_idx is not None:
                    arrival_label = arr_dt.strftime("%H:%M")
        except (ValueError, TypeError):
            pass

    if arrival_idx is not None and arrival_idx < len(temps):
        # Use forecast at arrival time
        code = codes[arrival_idx] if arrival_idx < len(codes) else 0
        forecast = []
        for i in range(arrival_idx, min(arrival_idx + 4, len(times))):
            forecast.append({
                "time": times[i][11:16] if len(times[i]) > 11 else times[i],
                "temp_c": temps[i] if i < len(temps) else None,
                "condition": _WMO_CODES.get(codes[i], "Unknown") if i < len(codes) else "Unknown",
                "icon": _WMO_ICONS.get(codes[i], "") if i < len(codes) else "",
                "rain_percent": precip[i] if i < len(precip) else None,
            })

        return WeatherInfo(
            location=name,
            temperature_c=temps[arrival_idx],
            feels_like_c=feels[arrival_idx] if arrival_idx < len(feels) else temps[arrival_idx],
            condition=_WMO_CODES.get(code, "Unknown"),
            icon=_WMO_ICONS.get(code, ""),
            wind_speed_kmh=winds[arrival_idx] if arrival_idx < len(winds) else 0,
            humidity_percent=int(humids[arrival_idx]) if arrival_idx < len(humids) else 0,
            forecast_hours=forecast,
            arrival_time=arrival_label,
        )

    # Fallback: use current conditions + next 4 hours
    current = data.get("current", {})
    code = current.get("weather_code", 0)

    now_dt = datetime.now(ZoneInfo("Europe/London"))
    now_str = now_dt.strftime("%Y-%m-%dT%H:00")
    start_idx = times.index(now_str) if now_str in times else 0

    forecast = []
    for i in range(start_idx, min(start_idx + 4, len(times))):
        forecast.append({
            "time": times[i][11:16] if len(times[i]) > 11 else times[i],
            "temp_c": temps[i] if i < len(temps) else None,
            "condition": _WMO_CODES.get(codes[i], "Unknown") if i < len(codes) else "Unknown",
            "icon": _WMO_ICONS.get(codes[i], "") if i < len(codes) else "",
            "rain_percent": precip[i] if i < len(precip) else None,
        })

    return WeatherInfo(
        location=name,
        temperature_c=current.get("temperature_2m", 0),
        feels_like_c=current.get("apparent_temperature", 0),
        condition=_WMO_CODES.get(code, "Unknown"),
        icon=_WMO_ICONS.get(code, ""),
        wind_speed_kmh=current.get("wind_speed_10m", 0),
        humidity_percent=int(current.get("relative_humidity_2m", 0)),
        forecast_hours=forecast,
    )


def format_weather_markdown(weather: WeatherInfo) -> str:
    """Format weather info as a compact markdown block for the chat UI."""
    if weather.arrival_time:
        header = (
            f"**{weather.icon} Weather at arrival ({weather.arrival_time}) in {weather.location}:** "
            f"{weather.condition}, {weather.temperature_c:.0f}°C "
            f"(feels like {weather.feels_like_c:.0f}°C)"
        )
    else:
        header = (
            f"**{weather.icon} Weather in {weather.location}:** "
            f"{weather.condition}, {weather.temperature_c:.0f}°C "
            f"(feels like {weather.feels_like_c:.0f}°C)"
        )
    lines = [
        header,
        f"Wind: {weather.wind_speed_kmh:.0f} km/h | Humidity: {weather.humidity_percent}%",
    ]
    if weather.forecast_hours:
        parts = []
        for h in weather.forecast_hours[:4]:
            rain = f" {h['rain_percent']}%☔" if h.get("rain_percent") and h["rain_percent"] > 20 else ""
            parts.append(f"{h['time']} {h['icon']} {h['temp_c']:.0f}°{rain}")
        lines.append("Forecast: " + " | ".join(parts))
    return "\n".join(lines)
