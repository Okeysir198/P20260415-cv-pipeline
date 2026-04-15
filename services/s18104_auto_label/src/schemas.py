"""Pydantic request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import DEFAULT_CONFIDENCE, DEFAULT_INCLUDE_MASKS, DEFAULT_MODE, DEFAULT_NMS_IOU, DEFAULT_OUTPUT_FORMAT


# ---------------------------------------------------------------------------
# Single image annotation
# ---------------------------------------------------------------------------


class AnnotateRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    classes: dict[int, str] = Field(..., description="Class mapping, e.g. {0: 'fire', 1: 'smoke'}")
    text_prompts: dict[str, str] = Field(
        default_factory=dict,
        description="Optional refined prompts per class, e.g. {'fire': 'flames and burning'}",
    )
    mode: str = Field(DEFAULT_MODE, description="Annotation mode: 'text' | 'auto' | 'hybrid'")
    confidence_threshold: float = Field(DEFAULT_CONFIDENCE, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(DEFAULT_NMS_IOU, ge=0.0, le=1.0)
    output_format: str = Field(DEFAULT_OUTPUT_FORMAT, description="'coco' | 'yolo' | 'yolo_seg' | 'label_studio'")
    include_masks: bool = Field(DEFAULT_INCLUDE_MASKS, description="Include base64 mask PNGs in response")

    # Optional: rule-based classification (detect intermediates, derive final classes)
    detection_classes: dict[str, str] | None = Field(
        None,
        description="Intermediate detection classes with prompts, e.g. {'head': 'a person head', 'helmet': 'hard hat'}. When set, 'classes' defines final output classes and 'detection_classes' defines what SAM3 actually detects.",
    )
    class_rules: list[dict] | None = Field(
        None,
        description="Rules to derive final classes from intermediate detections. Each rule: {output_class_id, source, condition (direct|overlap|no_overlap), target?, min_iou?}",
    )
    vlm_verify: dict | None = Field(
        None,
        description="VLM verification config: {model, ollama_url, verify_classes, priority, budget, vlm_min_confidence}",
    )


class Detection(BaseModel):
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: list[int] = Field(..., description="Pixel coords [x1, y1, x2, y2]")
    bbox_norm: list[float] = Field(..., description="Normalized [cx, cy, w, h]")
    polygon: list[list[float]] = Field(default_factory=list, description="Normalized [[x,y], ...] pairs")
    mask: str | None = Field(None, description="Base64 PNG mask (only if include_masks=True)")
    area: float = Field(0.0, description="Mask area as fraction of image")


class AnnotateResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int
    num_detections: int
    processing_time_s: float
    formatted_output: list = Field(default_factory=list, description="Output in requested format")


# ---------------------------------------------------------------------------
# Batch job
# ---------------------------------------------------------------------------


class ImageInput(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    filename: str = Field(..., description="Original filename for tracking")


class JobCreateRequest(BaseModel):
    images: list[ImageInput]
    classes: dict[int, str]
    text_prompts: dict[str, str] = Field(default_factory=dict)
    mode: str = Field(DEFAULT_MODE)
    confidence_threshold: float = Field(DEFAULT_CONFIDENCE, ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(DEFAULT_NMS_IOU, ge=0.0, le=1.0)
    output_format: str = Field(DEFAULT_OUTPUT_FORMAT)
    include_masks: bool = False
    webhook_url: str | None = None


class JobState(BaseModel):
    job_id: str
    status: str = Field(..., description="queued | running | completed | failed | cancelled")
    total_images: int
    processed_images: int = 0
    results: list[dict] = Field(default_factory=list)
    error: str | None = None
    created_at: float


class JobCreateResponse(BaseModel):
    job_id: str
    total_images: int
    status: str


class JobListItem(BaseModel):
    job_id: str
    status: str
    total_images: int
    processed_images: int
    created_at: float


# ---------------------------------------------------------------------------
# Video session
# ---------------------------------------------------------------------------


class VideoSessionRequest(BaseModel):
    mode: str = Field("tracker", description="SAM3 session mode: 'tracker' or 'video'")
    classes: dict[int, str] = Field(default_factory=dict)
    text: str | None = Field(None, description="Text prompt (required for video mode)")
    frames: list[str] | None = Field(None, description="Base64 frames (required for video mode)")
    output_format: str = Field(DEFAULT_OUTPUT_FORMAT)


class VideoSessionResponse(BaseModel):
    session_id: str
    sam3_session_id: str
    mode: str


class VideoFrameRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded frame")
    prompts: list[dict] | None = Field(None, description="Optional prompts for this frame")


class VideoFrameDetection(BaseModel):
    obj_id: int
    class_id: int = -1
    class_name: str = ""
    score: float
    bbox_xyxy: list[int]
    bbox_norm: list[float]
    polygon: list[list[float]] = Field(default_factory=list)
    mask: str | None = None
    area: float = 0.0


class VideoFrameResponse(BaseModel):
    frame_idx: int
    detections: list[VideoFrameDetection]


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


class ConvertRequest(BaseModel):
    detections: list[Detection]
    output_format: str
    image_width: int
    image_height: int


class ConvertResponse(BaseModel):
    formatted_output: list


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    sam3: str
    active_jobs: int
    active_video_sessions: int
