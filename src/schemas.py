"""Typed request model for the API (ranges come from src/config.py)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import RANGES


class PredictionInput(BaseModel):
    """One farm record. Wrong type, missing field or out-of-range value -> HTTP 422."""
    model_config = ConfigDict(extra="forbid")
    rainfall_mm: float = Field(ge=RANGES["rainfall_mm"][0], le=RANGES["rainfall_mm"][1])
    avg_temp_c: float = Field(ge=RANGES["avg_temp_c"][0], le=RANGES["avg_temp_c"][1])
    soil_moisture_pct: float = Field(ge=RANGES["soil_moisture_pct"][0], le=RANGES["soil_moisture_pct"][1])
    fertilizer_kg: float = Field(ge=RANGES["fertilizer_kg"][0], le=RANGES["fertilizer_kg"][1])
    field_size_ha: float = Field(ge=RANGES["field_size_ha"][0], le=RANGES["field_size_ha"][1])
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]
