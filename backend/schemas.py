from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResultItem(BaseModel):
    id: str
    title: str
    species: str
    score: float
    recording_id: str | None = None
    scientific_name: str | None = None
    vocalization_type: str | None = None
    duration: str | None = None
    image_url: str | None = None
    audio_url: str | None = None
    species_code: str | None = None
    species_description: str | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResultItem]


class ClassificationHit(BaseModel):
    label: str
    scientificName: str | None = None
    score: float


class ClassificationResponse(BaseModel):
    count: int
    results: list[ClassificationHit]


class DescribeAudioResponse(BaseModel):
    descriptions: list[str]
