"""Annotation schema definitions for image annotation system.

This module defines Pydantic v2 models for:
- PrimitiveEntity: Geometric primitives (lines, circles, contours)
- SemanticLabel: GPT-4o-mini output (restricted to labeling only)
- Relation: Relationships between entities
- AnnotationGraph: Complete annotation structure
- Claim: Failure-aware reasoning claims with dependencies
- Contradiction: Detected contradictions in annotations

All models are designed to prevent hallucinations by requiring pixel coordinates
and explicit uncertainty flags.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
import uuid


class PrimitiveEntity(BaseModel):
    """Base class for geometric primitives extracted from images.
    
    Attributes:
        id: Unique identifier for the primitive
        pixel_coords: List of pixel coordinates (required for all primitives)
        primitive_type: Type of primitive (line, circle, or contour)
        metadata: Optional metadata (e.g., confidence from OpenCV)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pixel_coords: List[Tuple[int, int]] = Field(
        description="List of pixel coordinates. For lines: [start, end]. For circles: [center]. For contours: all points."
    )
    primitive_type: Literal["line", "circle", "contour", "intersection"] = Field(
        description="Type of geometric primitive"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata from extraction")
    
    @field_validator('pixel_coords')
    @classmethod
    def validate_pixel_coords(cls, v: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Ensure pixel coordinates are non-empty."""
        if not v:
            raise ValueError("pixel_coords cannot be empty")
        return v


class Line(BaseModel):
    """Line primitive with start and end points.
    
    Attributes:
        start_point: Starting pixel coordinates (x, y)
        end_point: Ending pixel coordinates (x, y)
        entity_id: Reference to PrimitiveEntity if part of graph
    """
    start_point: Tuple[int, int] = Field(description="Starting pixel coordinates (x, y)")
    end_point: Tuple[int, int] = Field(description="Ending pixel coordinates (x, y)")
    entity_id: Optional[str] = None


class Circle(BaseModel):
    """Circle primitive with center and radius.
    
    Attributes:
        center: Center pixel coordinates (x, y)
        radius: Radius in pixels
        entity_id: Reference to PrimitiveEntity if part of graph
    """
    center: Tuple[int, int] = Field(description="Center pixel coordinates (x, y)")
    radius: float = Field(gt=0, description="Radius in pixels")
    entity_id: Optional[str] = None


class Contour(BaseModel):
    """Contour primitive as a polygon.
    
    Attributes:
        points: List of pixel coordinates forming the polygon
        entity_id: Reference to PrimitiveEntity if part of graph
    """
    points: List[Tuple[int, int]] = Field(description="List of pixel coordinates forming the polygon")
    entity_id: Optional[str] = None


class Intersection(BaseModel):
    """Intersection point between primitives.
    
    Attributes:
        point: Intersection pixel coordinates (x, y)
        primitive_ids: List of primitive IDs that intersect at this point
        entity_id: Reference to PrimitiveEntity if part of graph
    """
    point: Tuple[int, int] = Field(description="Intersection pixel coordinates (x, y)")
    primitive_ids: List[str] = Field(description="List of primitive IDs that intersect")
    entity_id: Optional[str] = None


class SemanticLabel(BaseModel):
    """GPT-4o-mini output for labeling primitives (restricted).
    
    Attributes:
        entity_id: Reference to PrimitiveEntity (MUST exist in provided primitives)
        label: Semantic label text
        uncertainty: Uncertainty score (0-1, where 1 is most uncertain)
        tentative: Whether this label is tentative (flagged by GPT)
        reasoning: Optional reasoning for the label
    """
    entity_id: str = Field(description="Reference to PrimitiveEntity.id")
    label: str = Field(description="Semantic label text")
    uncertainty: float = Field(ge=0.0, le=1.0, description="Uncertainty score (0=certain, 1=uncertain)")
    tentative: bool = Field(default=False, description="Whether label is tentative")
    reasoning: Optional[str] = Field(default=None, description="Optional reasoning for the label")


class Relation(BaseModel):
    """Relationship between entities.
    
    Attributes:
        source_id: Source entity ID
        target_id: Target entity ID
        relation_type: Type of relation (e.g., "parallel", "perpendicular", "contains")
        uncertainty: Uncertainty score (0-1)
        tentative: Whether relation is tentative
    """
    source_id: str = Field(description="Source entity ID")
    target_id: str = Field(description="Target entity ID")
    relation_type: str = Field(description="Type of relation")
    uncertainty: float = Field(ge=0.0, le=1.0, description="Uncertainty score (0=certain, 1=uncertain)")
    tentative: bool = Field(default=True, description="Relations are tentative by default")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional relation metadata")


class Contradiction(BaseModel):
    """Detected contradiction in annotations.
    
    Attributes:
        contradiction_type: Type of contradiction (cycle, mutually_exclusive, unsupported)
        entities_involved: List of entity IDs involved in contradiction
        description: Human-readable description of the contradiction
        severity: Severity level (low, medium, high)
    """
    contradiction_type: Literal["cycle", "mutually_exclusive", "unsupported", "geometry_inconsistent"] = Field(
        description="Type of contradiction detected"
    )
    entities_involved: List[str] = Field(description="List of entity IDs involved")
    description: str = Field(description="Human-readable description")
    severity: Literal["low", "medium", "high"] = Field(default="medium", description="Severity level")


class Claim(BaseModel):
    """Claim for failure-aware reasoning with dependency tracking.
    
    Attributes:
        claim_id: Unique identifier for the claim
        annotation: The SemanticLabel this claim represents
        dependencies: List of other claim IDs this claim depends on
        invalidated_by: Claim ID that invalidates this claim (if any)
        contradiction_with: Claim ID this contradicts (if any)
        robustness_score: Computed robustness score (0-1)
    """
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    annotation: SemanticLabel = Field(description="The semantic label this claim represents")
    dependencies: List[str] = Field(default_factory=list, description="List of claim IDs this depends on")
    invalidated_by: Optional[str] = Field(default=None, description="Claim ID that invalidates this")
    contradiction_with: Optional[str] = Field(default=None, description="Claim ID this contradicts")
    robustness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Robustness score (1=robust, 0=fragile)")


class AnnotationGraph(BaseModel):
    """Complete annotation structure with entities, labels, relations, and contradictions.
    
    Attributes:
        entities: List of detected primitives
        labels: List of semantic labels from GPT-4o-mini
        relations: List of relations between entities
        contradictions: List of detected contradictions
        metadata: Optional metadata (e.g., image dimensions, processing time)
    """
    entities: List[PrimitiveEntity] = Field(default_factory=list, description="List of detected primitives")
    labels: List[SemanticLabel] = Field(default_factory=list, description="List of semantic labels")
    relations: List[Relation] = Field(default_factory=list, description="List of relations")
    contradictions: List[Contradiction] = Field(default_factory=list, description="List of contradictions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    
    def get_entity_by_id(self, entity_id: str) -> Optional[PrimitiveEntity]:
        """Get entity by ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None
    
    def get_labels_for_entity(self, entity_id: str) -> List[SemanticLabel]:
        """Get all labels for a specific entity."""
        return [label for label in self.labels if label.entity_id == entity_id]
    
    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving a specific entity."""
        return [
            rel for rel in self.relations
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]


class AnnotationResult(BaseModel):
    """Result of annotation process.
    
    Attributes:
        annotation_graph: The complete annotation graph
        overlay_image: Annotated image with overlays (as numpy array, serialized)
        formal_report: Structured text report
        json_output: JSON representation of annotations
        instability_detected: Whether temporal instability was detected
        instability_reason: Reason for instability (if any)
        processing_time: Processing time in seconds
    """
    annotation_graph: AnnotationGraph = Field(description="Complete annotation graph")
    overlay_image: Optional[bytes] = Field(default=None, description="Annotated image (serialized)")
    formal_report: str = Field(description="Structured text report")
    json_output: Dict[str, Any] = Field(description="JSON representation")
    instability_detected: bool = Field(default=False, description="Whether temporal instability detected")
    instability_reason: Optional[str] = Field(default=None, description="Reason for instability")
    processing_time: float = Field(description="Processing time in seconds")
    frame_id: Optional[int] = Field(default=None, description="Frame ID if part of sequence")
