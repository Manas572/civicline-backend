from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


IssueType = Literal["pothole", "water_leak", "garbage", "streetlight", "bridge_crack", "other", "not_civic"]
ImageQuality = Literal["good", "poor", "unusable"]
Consistency = Literal["yes", "no", "unclear", "no_report"]
PotholeSize = Literal["small", "medium", "large", "unknown"]
CrackExtent = Literal["short", "long", "structural", "unknown"]
WaterPooling = Literal["none", "minor", "extensive", "unknown"]


class VisionObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    issue_type: IssueType
    visual_evidence: list[str] = Field(default_factory=list)
    visual_risk_factors: list[str] = Field(default_factory=list)
    pothole_size: PotholeSize = "unknown"
    crack_extent: CrackExtent = "unknown"
    water_pooling: WaterPooling = "unknown"
    suggested_severity: int = Field(ge=1, le=5)
    justification: str
    hazards: list[str] = Field(default_factory=list)
    image_quality: ImageQuality
    consistent_with_citizen_report: Consistency = "no_report"
    model_confidence_uncalibrated: float | None = None


class TicketStatus(str, Enum):
    awaiting_review = "awaiting_review"
    submitted = "submitted"
    assigned = "assigned"
    in_progress = "in_progress"
    resolved = "resolved"
    reopened = "reopened"


class TicketPatch(BaseModel):
    issue_type: IssueType | None = None
    severity: int | None = Field(default=None, ge=1, le=5)
    department: str | None = Field(default=None, min_length=1, max_length=80)
    status: TicketStatus | None = None
    reason: str = Field(min_length=3, max_length=1000)
    actor: str = Field(default="staff", min_length=1, max_length=120)


class CommentCreate(BaseModel):
    author_label: str = Field(default="Citizen", min_length=1, max_length=60)
    body: str = Field(min_length=2, max_length=1200)


class VoteCreate(BaseModel):
    pass


class RatingCreate(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)


class RoutingRuleCreate(BaseModel):
    issue_type: IssueType | None = None
    ward_id: str | None = Field(default=None, max_length=80)
    department: str = Field(min_length=1, max_length=100)
    officer: str | None = Field(default=None, max_length=120)
    queue: str | None = Field(default=None, max_length=120)
    sla_hours: float = Field(gt=0, le=8760)
    escalation_path: list[str] = Field(default_factory=lambda: ["JE", "AE", "EE", "Commissioner"])
    priority: int = Field(default=100, ge=1, le=10000)
    active: bool = True
