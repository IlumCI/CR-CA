"""Image Annotation module for CR-CA.

This module provides live image annotation with GPT-4o-mini under adversarial constraints.
"""

from image_annotation.annotation_engine import ImageAnnotationEngine

from schemas.annotation import (
    AnnotationResult,
    AnnotationGraph,
    PrimitiveEntity,
    SemanticLabel,
    Relation,
    Contradiction,
    Claim
)

__all__ = [
    "ImageAnnotationEngine",
    "AnnotationResult",
    "AnnotationGraph",
    "PrimitiveEntity",
    "SemanticLabel",
    "Relation",
    "Contradiction",
    "Claim"
]
