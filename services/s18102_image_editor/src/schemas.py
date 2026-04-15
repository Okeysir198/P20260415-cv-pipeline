"""Pydantic request/response schemas for all Image Editor endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InpaintRequest(BaseModel):
    """Request for image inpainting using SAM3 + Flux NIM."""

    image: str = Field(..., description="Base64-encoded input image")
    prompt: str = Field(..., description="Text prompt describing the desired edit")
    mask: Optional[str] = Field(None, description="Base64-encoded mask (white = inpaint region)")
    bbox: Optional[list[float]] = Field(None, description="Bounding box [x1,y1,x2,y2] → SAM3 segment_box")
    text_prompt: Optional[str] = Field(None, description="Text prompt → SAM3 segment_text")
    num_variants: int = Field(1, ge=1, le=4, description="Number of output variants")
    seed: Optional[int] = Field(0, description="Random seed")
    steps: int = Field(4, ge=1, le=50, description="Inference steps for Flux")


class InpaintResponse(BaseModel):
    """Response from inpaint endpoint."""

    images: list[str] = Field(..., description="Base64-encoded result images")
    mask_used: Optional[str] = Field(None, description="Base64 mask used (None if direct edit)")
    seed: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    flux_nim: str
    sam3: str
