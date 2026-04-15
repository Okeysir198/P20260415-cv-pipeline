"""Pydantic request/response schemas for all SAM3 endpoints."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --- Image endpoints ---

class BoxRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    box: list[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixels")

    @field_validator("box")
    @classmethod
    def validate_box(cls, v):
        x1, y1, x2, y2 = v
        if any(c < 0 for c in v):
            raise ValueError("Box coordinates must be non-negative")
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid box: x1({x1}) must be < x2({x2}), y1({y1}) must be < y2({y2})")
        return v


class BoxResultData(BaseModel):
    mask: str = Field(..., description="Base64 grayscale PNG")
    bbox: dict
    score: float
    iou_score: float
    area: float = 0.0


class BoxResponse(BaseModel):
    result: BoxResultData


class TextRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    text: str = Field(..., description="Text prompt for open-vocab segmentation")
    detection_threshold: Optional[float] = None
    mask_threshold: Optional[float] = None


class Detection(BaseModel):
    mask: str = Field(..., description="Base64 grayscale PNG")
    bbox: dict
    score: float
    area: float = 0.0


class TextResponse(BaseModel):
    detections: list[Detection]


class AutoMaskRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    threshold: Optional[float] = None
    prompts: Optional[list[str]] = None


class AutoMaskResponse(BaseModel):
    detections: list[Detection]


# --- Batch endpoints (true tensor batching) ---


class BatchTextItem(BaseModel):
    """Single image for batch text segmentation (text is provided at request level)."""
    image: str = Field(..., description="Base64-encoded image")


class BatchTextRequest(BaseModel):
    items: list[BatchTextItem] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="List of image requests (all use the same text prompt)",
    )
    text: str = Field(..., description="Text prompt for all images")
    detection_threshold: Optional[float] = None
    mask_threshold: Optional[float] = None


class BatchAutoMaskItem(BaseModel):
    """Single image for batch auto mask (threshold/prompts are provided at request level)."""
    image: str = Field(..., description="Base64-encoded image")


class BatchAutoMaskRequest(BaseModel):
    items: list[BatchAutoMaskItem] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="List of auto-mask requests",
    )
    threshold: Optional[float] = None
    prompts: Optional[list[str]] = None


# --- Session endpoints ---

class SessionCreateRequest(BaseModel):
    mode: Literal["tracker", "video"] = Field(..., description="'tracker' or 'video'")
    frames: Optional[list[str]] = Field(None, max_length=100, description="Base64 frames (file mode)")
    text: Optional[str] = Field(None, description="Text prompt (video mode)")


class SessionCreateResponse(BaseModel):
    session_id: str
    mode: str
    num_frames: int
    width: int
    height: int


class FrameAddRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded frame")


class SessionDetection(BaseModel):
    obj_id: int
    mask: str
    bbox: dict
    score: float
    area: float = 0.0


class FrameResponse(BaseModel):
    frame_idx: int
    detections: list[SessionDetection]


class PromptRequest(BaseModel):
    frame_idx: int = Field(..., ge=0)
    obj_ids: list[int] = Field(..., min_length=1)
    points: Optional[list] = None
    labels: Optional[list] = None
    boxes: Optional[list] = None
    masks: Optional[list] = None


class PropagateRequest(BaseModel):
    max_frames: Optional[int] = None


class PropagateResponse(BaseModel):
    frames: list[FrameResponse]


class HealthResponse(BaseModel):
    status: str
    device: str
    model: str
    dtype: str
    loaded: dict
    sessions: dict
