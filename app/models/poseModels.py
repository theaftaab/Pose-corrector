from typing import Literal

from pydantic import BaseModel

ViewType = Literal["front", "side", "ambiguous", "unknown"]
StatusType = Literal["ok", "warn", "bad"]


class PostureMetric(BaseModel):
    region: str
    value: float
    unit: str
    target: str
    status: StatusType


class PersonAnalysis(BaseModel):
    person_id: int
    view: ViewType
    person_detected: bool
    metrics: list[PostureMetric]


class PostureAnalysisResponse(BaseModel):
    people: list[PersonAnalysis]
    image_base64: str


__all__ = [
    "PostureMetric",
    "PersonAnalysis",
    "PostureAnalysisResponse",
]
