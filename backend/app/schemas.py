"""
Request/response shapes for the /predict endpoint, matching
INPUT_OUTPUT_SPEC.md exactly.

Deliberate design choice: PatientInput does NOT enforce the valid ranges
(e.g. age between 1-120) here with Pydantic's ge=/le= constraints. If it
did, an out-of-range value would trigger FastAPI's generic 422 error
format instead of the friendly {"error": true, "messages": [...]} shape
that predict_core.py was specifically designed to produce. So Pydantic
only checks "is this actually a number" (basic type safety at the API
boundary), and predict_core.validate_input() -- which Track A already
wrote and documented -- remains the single source of truth for range
validation and its error messages.
"""
from typing import List, Optional

from pydantic import BaseModel


class PatientInput(BaseModel):
    age: float
    bmi: float
    bp_diastolic: float
    glucose_mgdl: float
    cholesterol_cat: float


class Driver(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    error: bool
    diabetes_risk_probability: Optional[float] = None
    diabetes_risk_label: Optional[int] = None
    cvd_risk_probability: Optional[float] = None
    cvd_risk_label: Optional[int] = None
    top_diabetes_drivers: Optional[List[Driver]] = None
    top_cvd_drivers: Optional[List[Driver]] = None
    messages: Optional[List[str]] = None
