"""Pydantic request/response models for the predict endpoint."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Report-time inputs. Only location + start time are required; everything
    else is optional and improves the prediction when provided."""

    # Required
    latitude: float = Field(..., description="Bengaluru latitude (12.6-13.4)")
    longitude: float = Field(..., description="Bengaluru longitude (77.2-77.9)")
    start_datetime: str = Field(..., description="ISO timestamp when the event starts")

    # Strongly recommended
    event_cause: Optional[str] = None
    event_type: Optional[str] = "unplanned"
    description: Optional[str] = ""

    # Optional categorical context
    police_station: Optional[str] = None
    corridor: Optional[str] = None
    zone: Optional[str] = None
    junction: Optional[str] = None
    direction: Optional[str] = None
    veh_type: Optional[str] = None
    reason_breakdown: Optional[str] = None
    cargo_material: Optional[str] = None
    authenticated: Optional[str] = "yes"
    address: Optional[str] = None

    # Optional numeric / timing
    age_of_truck: Optional[float] = None
    created_date: Optional[str] = Field(
        None, description="When the event was reported (advance notice). Defaults to start time."
    )

    def to_payload(self) -> dict:
        return self.model_dump(exclude_none=False)


class PredictResponse(BaseModel):
    # Hero output
    manpower_level: str
    manpower_tier: str
    officers_suggested: int
    # Closure
    closure_probability: float
    closure_expected: bool
    # Priority
    high_priority_probability: float
    # Duration
    expected_duration_min: float
    duration_low_min: float
    duration_high_min: float
    # Hotspot
    hotspot_risk: Optional[float] = None
    hotspot_flag: int = 0
    # Recommendations
    barricading: str
    diversion: str
    rationale: str
    # Echo
    event_cause: Optional[str] = None
    event_type: Optional[str] = None
    address: Optional[str] = None
