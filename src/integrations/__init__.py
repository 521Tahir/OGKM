"""External integrations for the Jeju Tourism Ontology system.

Currently provides:
  - KMA (Korea Meteorological Administration) weather client with offline fallback.
"""
from .weather_client import KMAWeatherClient, WeatherForecast

__all__ = ["KMAWeatherClient", "WeatherForecast"]
