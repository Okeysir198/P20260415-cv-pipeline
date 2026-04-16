"""Pydantic models for the Annotation QA Service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParsedAnnotation(BaseModel):
    class_id: int
    bbox_norm: list[float] = Field(..., description="[cx, cy, w, h] normalized")
    polygon_norm: list[float] = Field(default_factory=list, description="[x1,y1,x2,y2,...] normalized")
    source_format: str = Field(..., description="yolo | yolo_seg | coco")


class ValidationIssue(BaseModel):
    type: str = Field(..., description="out_of_bounds, invalid_class, degenerate_box, etc.")
    severity: str = Field(..., description="high | medium | low")
    annotation_idx: int | None = None
    detail: str = ""


class SuggestedFix(BaseModel):
    type: str = Field(..., description="clip_bbox, remove_duplicate, remove_degenerate, tighten_bbox, remove_annotation")
    annotation_idx: int
    original: dict | None = None
    suggested: dict | None = None
    reason: str = ""


class ValidateRequest(BaseModel):
    image: str = Field(..., description="Base64 image (for dimensions)")
    labels: list[str] | list[dict] = Field(..., description="YOLO strings OR COCO dicts")
    label_format: str = Field("yolo", description="yolo | yolo_seg | coco")
    classes: dict[int, str] = Field(..., description="{0: 'fire', 1: 'smoke'}")
    config: dict = Field(default_factory=dict, description="Optional threshold overrides")


class ValidateResponse(BaseModel):
    issues: list[ValidationIssue]
    num_annotations: int
    num_issues: int
    quality_score: float
    grade: str
    suggested_fixes: list[SuggestedFix]
    label_format: str
    processing_time_s: float


class VerifyRequest(BaseModel):
    image: str
    labels: list[str] | list[dict]
    label_format: str = "yolo"
    classes: dict[int, str]
    text_prompts: dict[str, str] = Field(default_factory=dict)
    include_missing_detection: bool = True
    config: dict = Field(default_factory=dict)
    enable_vlm: bool = False
    vlm_trigger: Literal["all", "selective", "standalone"] = "selective"
    class_rules: list[dict] = Field(default_factory=list, description="Rules that derived the labels (direct/overlap/no_overlap)")
    vlm_budget: dict = Field(default_factory=dict, description="VLM priority sampling budget: {sample_rate, max_samples, priority}")


class SAM3Verification(BaseModel):
    box_ious: list[float] | None  # None means SAM3 was unavailable for all annotations
    mask_ious: list[float] = Field(default_factory=list)
    mean_box_iou: float = 0.0
    mean_mask_iou: float = 0.0
    misclassified: list[int] = Field(default_factory=list)
    missing_detections: list[dict] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    issues: list[ValidationIssue]
    sam3_verification: SAM3Verification
    num_annotations: int
    num_issues: int
    quality_score: float | None  # None when grade is 'unverified' (SAM3 unavailable)
    grade: str
    suggested_fixes: list[SuggestedFix]
    label_format: str
    processing_time_s: float
    vlm_verification: VLMVerification | None = None


class FixRequest(BaseModel):
    labels: list[str] | list[dict]
    label_format: str = "yolo"
    classes: dict[int, str]
    issues: list[dict] = Field(default_factory=list)
    suggested_fixes: list[dict] = Field(default_factory=list)
    auto_apply: list[str] = Field(
        default_factory=lambda: ["clip_bbox", "remove_duplicate", "remove_degenerate"],
    )


class FixResponse(BaseModel):
    corrected_labels: list[str] | list[dict]
    applied_fixes: list[dict]
    needs_review: list[dict]
    num_applied: int
    num_needs_review: int
    num_annotations_before: int
    num_annotations_after: int


class ImageWithLabels(BaseModel):
    image: str
    labels: list[str] | list[dict]
    filename: str


class QAJobRequest(BaseModel):
    images: list[ImageWithLabels]
    label_format: str = "yolo"
    classes: dict[int, str]
    text_prompts: dict[str, str] = Field(default_factory=dict)
    mode: str = Field("verify", description="validate | verify")
    include_missing_detection: bool = False
    config: dict = Field(default_factory=dict, description="Optional threshold overrides")
    webhook_url: str | None = None
    enable_vlm: bool = False
    vlm_trigger: Literal["all", "selective", "standalone"] = "selective"
    class_rules: list[dict] = Field(default_factory=list, description="Rules that derived the labels")
    vlm_budget: dict = Field(default_factory=dict, description="VLM priority sampling budget")


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


class ReportRequest(BaseModel):
    results: list[dict]
    dataset_name: str = "unknown"
    classes: dict[int, str] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    dataset: str
    total_checked: int
    grades: dict[str, int]
    avg_quality_score: float
    issue_breakdown: dict[str, int]
    per_class_stats: dict[str, dict]
    worst_images: list[dict]
    auto_fixable_count: int


class VLMCropResult(BaseModel):
    annotation_idx: int
    class_name: str
    is_correct: bool
    confidence: float = 0.0
    reason: str = ""


class VLMCropVerification(BaseModel):
    results: list[VLMCropResult] = Field(default_factory=list)
    num_checked: int = 0
    num_incorrect: int = 0
    mean_confidence: float = 0.0


class VLMSceneVerification(BaseModel):
    incorrect_indices: list[int] = Field(default_factory=list)
    missing_descriptions: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    raw_response: str = ""


class VLMVerification(BaseModel):
    crop_verification: VLMCropVerification = Field(default_factory=VLMCropVerification)
    scene_verification: VLMSceneVerification = Field(default_factory=VLMSceneVerification)
    available: bool = False


class HealthResponse(BaseModel):
    status: str
    sam3: str
    ollama: str = "unknown"
    active_jobs: int
