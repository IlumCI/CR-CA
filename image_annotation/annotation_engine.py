"""Image Annotation Engine with GPT-4o-mini under adversarial constraints.

This module implements a comprehensive image annotation system designed to neutralize
GPT-4o-mini's failure modes through adversarial containment. The system pre-processes
images, extracts geometric primitives via OpenCV, restricts GPT-4o-mini to semantic
labeling only, compiles annotations into a typed graph with contradiction detection,
uses deterministic math, integrates with CR-CA for failure-aware reasoning, tracks
temporal coherence, and outputs overlay, formal report, and JSON.
"""

import base64
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from loguru import logger
import rustworkx as rx
from PIL import Image, ImageDraw, ImageFont

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available, progress bars will be disabled")

# Try to import yaml for config files
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Try to import requests for URL loading
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available, URL loading will be disabled")

# Optional dependencies with graceful fallbacks
try:
    from skimage import exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available, histogram equalization will be disabled")

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    logger.warning("pywavelets not available, noise-aware downscaling will be disabled")

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("sympy not available, symbolic math will be disabled")

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("z3-solver not available, constraint solving will be disabled")

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    logger.warning("filterpy not available, temporal tracking will be disabled")

# Local imports
from CRCA import CRCAAgent
from schemas.annotation import (
    PrimitiveEntity,
    Line,
    Circle,
    Contour,
    Intersection,
    SemanticLabel,
    Relation,
    Contradiction,
    Claim,
    AnnotationGraph,
    AnnotationResult
)
from prompts.image_annotation import RESTRICTED_LABELER_SYSTEM_PROMPT


@dataclass
class AnnotationConfig:
    """Configuration for image annotation engine.
    
    All parameters are optional - None means auto-tune/auto-detect.
    """
    # Model settings
    gpt_model: str = "gpt-4o-mini"
    use_crca_tools: bool = False
    
    # Feature toggles
    enable_temporal_tracking: bool = True
    cache_enabled: bool = True
    auto_retry: bool = True
    auto_tune_params: bool = True
    auto_detect_type: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_backoff: float = 1.5  # Exponential backoff multiplier
    
    # Output settings
    output_format: str = "overlay"  # "overlay", "json", "report", "all"
    
    # Auto-tuned parameters (None = auto-tune)
    opencv_params: Optional[Dict[str, Any]] = None
    preprocessing_params: Optional[Dict[str, Any]] = None
    
    # Batch processing
    parallel_workers: Optional[int] = None  # None = auto-detect
    show_progress: bool = True
    
    # Cache settings
    cache_dir: Optional[str] = None  # None = use default .cache directory
    
    @classmethod
    def from_env(cls) -> "AnnotationConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Load from environment
        if os.getenv("ANNOTATION_GPT_MODEL"):
            config.gpt_model = os.getenv("ANNOTATION_GPT_MODEL")
        if os.getenv("ANNOTATION_CACHE_ENABLED"):
            config.cache_enabled = os.getenv("ANNOTATION_CACHE_ENABLED").lower() == "true"
        if os.getenv("ANNOTATION_AUTO_RETRY"):
            config.auto_retry = os.getenv("ANNOTATION_AUTO_RETRY").lower() == "true"
        if os.getenv("ANNOTATION_MAX_RETRIES"):
            config.max_retries = int(os.getenv("ANNOTATION_MAX_RETRIES"))
        if os.getenv("ANNOTATION_OUTPUT_FORMAT"):
            config.output_format = os.getenv("ANNOTATION_OUTPUT_FORMAT")
        if os.getenv("ANNOTATION_CACHE_DIR"):
            config.cache_dir = os.getenv("ANNOTATION_CACHE_DIR")
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "AnnotationConfig":
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    logger.warning("PyYAML not available, cannot load YAML config")
                    return cls()
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    data = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_path.suffix}")
                return cls()
            
            # Create config from dict
            return cls(**data)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return cls()


class ImageAnnotationEngine:
    """Main god-class for image annotation with adversarial constraints.
    
    This class orchestrates the entire annotation pipeline:
    1. Image preprocessing (reduce entropy)
    2. Primitive extraction (OpenCV)
    3. Semantic labeling (GPT-4o-mini, restricted)
    4. Graph compilation (rustworkx)
    5. Contradiction detection
    6. Deterministic math computations
    7. Temporal tracking (Kalman filters)
    8. CR-CA integration (failure-aware reasoning)
    9. Output generation (overlay, report, JSON)
    
    Attributes:
        gpt_model: GPT model name (default: "gpt-4o-mini")
        enable_temporal_tracking: Whether to enable temporal tracking
        _labeler: CRCAAgent instance for labeling
        _crca_agent: CRCAAgent instance for failure-aware reasoning
        _entity_trackers: Dict mapping entity IDs to Kalman filters
        _frame_history: List of previous frame annotations
        _claims: Dict mapping claim IDs to Claim objects
    """
    
    def __init__(
        self,
        config: Optional[AnnotationConfig] = None,
        gpt_model: Optional[str] = None,
        enable_temporal_tracking: Optional[bool] = None,
        use_crca_tools: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
        auto_retry: Optional[bool] = None,
        output_format: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the annotation engine.
        
        Args:
            config: Optional AnnotationConfig (if None, loads from env/file/defaults)
            gpt_model: GPT model name (overrides config)
            enable_temporal_tracking: Enable temporal tracking (overrides config)
            use_crca_tools: Whether to enable CRCA tools (overrides config)
            cache_enabled: Enable caching (overrides config)
            auto_retry: Enable auto-retry (overrides config)
            output_format: Output format (overrides config)
            **kwargs: Additional arguments passed to CRCAAgent
        """
        # Load configuration (priority: explicit params > config > env > defaults)
        if config is None:
            config = self._load_config()
        
        # Override with explicit parameters if provided
        if gpt_model is not None:
            config.gpt_model = gpt_model
        if enable_temporal_tracking is not None:
            config.enable_temporal_tracking = enable_temporal_tracking
        if use_crca_tools is not None:
            config.use_crca_tools = use_crca_tools
        if cache_enabled is not None:
            config.cache_enabled = cache_enabled
        if auto_retry is not None:
            config.auto_retry = auto_retry
        if output_format is not None:
            config.output_format = output_format
        
        self.config = config
        self.gpt_model = config.gpt_model
        self.enable_temporal_tracking = config.enable_temporal_tracking and FILTERPY_AVAILABLE
        
        # Initialize labeling agent (restricted GPT-4o-mini)
        self._labeler = CRCAAgent(
            model_name=config.gpt_model,
            system_prompt=RESTRICTED_LABELER_SYSTEM_PROMPT,
            agent_name="image-annotation-labeler",
            agent_description="Restricted semantic labeler for image annotations",
            use_crca_tools=config.use_crca_tools,
            agent_max_loops=1,  # Single pass for labeling
            **kwargs
        )
        
        # Initialize CR-CA agent for failure-aware reasoning
        self._crca_agent = CRCAAgent(
            model_name=config.gpt_model,
            agent_name="image-annotation-crca",
            agent_description="Failure-aware reasoning for image annotations",
            use_crca_tools=True,
            **kwargs
        )
        
        # Temporal tracking state
        self._entity_trackers: Dict[str, Any] = {}  # entity_id -> KalmanFilter
        self._frame_history: List[AnnotationGraph] = []
        self._entity_id_map: Dict[str, str] = {}  # Maps entity IDs across frames
        self._claims: Dict[str, Claim] = {}
        
        # Cache setup
        self._cache: Dict[str, Any] = {}  # cache_key -> cached_data
        if config.cache_dir:
            self._cache_dir = Path(config.cache_dir)
        else:
            self._cache_dir = Path.home() / ".cache" / "image_annotation"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ImageAnnotationEngine initialized with model={config.gpt_model}, temporal_tracking={self.enable_temporal_tracking}, cache={config.cache_enabled}")
    
    def _load_config(self) -> AnnotationConfig:
        """Load configuration from environment variables, config file, or defaults."""
        # Try environment variables first
        config = AnnotationConfig.from_env()
        
        # Try config file (config.yaml or config.json in current directory or home)
        config_paths = [
            Path("config.yaml"),
            Path("config.json"),
            Path.home() / ".image_annotation_config.yaml",
            Path.home() / ".image_annotation_config.json"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                file_config = AnnotationConfig.from_file(config_path)
                # Merge: file config overrides env config
                for key, value in file_config.__dict__.items():
                    if value is not None and value != AnnotationConfig().__dict__[key]:
                        setattr(config, key, value)
                break
        
        return config
    
    # ==================== Smart Input Handling ====================
    
    def _detect_input_type(self, input: Any) -> str:
        """Detect the type of input."""
        if isinstance(input, str):
            if input.startswith(('http://', 'https://')):
                return 'url'
            return 'file_path'
        elif isinstance(input, (Path, os.PathLike)):
            return 'file_path'
        elif isinstance(input, np.ndarray):
            return 'numpy_array'
        elif isinstance(input, Image.Image):
            return 'pil_image'
        elif isinstance(input, list):
            return 'batch'
        else:
            return 'unknown'
    
    def _auto_load_input(self, input: Any) -> np.ndarray:
        """
        Automatically load and convert input to numpy array.
        
        Supports: file paths, URLs, numpy arrays, PIL Images
        """
        input_type = self._detect_input_type(input)
        
        if input_type == 'numpy_array':
            return input.copy()
        
        elif input_type == 'file_path':
            path = Path(input) if not isinstance(input, Path) else input
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not load image from: {path}")
            return image
        
        elif input_type == 'url':
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests library required for URL loading. Install with: pip install requests")
            response = requests.get(input, timeout=30)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image from URL: {input}")
            return image
        
        elif input_type == 'pil_image':
            # Convert PIL to numpy
            if input.mode != 'RGB':
                input = input.convert('RGB')
            image_array = np.array(input)
            # PIL uses RGB, OpenCV uses BGR
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            return image
        
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    def _convert_to_numpy(self, input: Any) -> np.ndarray:
        """Convert any input to numpy array (alias for _auto_load_input)."""
        return self._auto_load_input(input)
    
    # ==================== Image Type Detection ====================
    
    def _detect_image_type(self, image: np.ndarray) -> str:
        """
        Automatically detect image type based on visual characteristics.
        
        Returns: "circuit", "architectural", "mathematical", "technical", "general"
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        total_pixels = h * w
        
        # Analyze image characteristics
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        line_density = line_count / (total_pixels / 10000)  # Normalize
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=min(h, w) // 4
        )
        circle_count = len(circles[0]) if circles is not None else 0
        
        # Detect contours (closed shapes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        
        # Color analysis (for circuit diagrams - often have colored components)
        if len(image.shape) == 3:
            color_variance = np.std(image, axis=2).mean()
        else:
            color_variance = 0
        
        # Heuristic rules for image type detection
        # Circuit diagrams: high line density, many circles (components), moderate contours
        if line_density > 0.5 and circle_count > 5 and 10 < contour_count < 100:
            return "circuit"
        
        # Architectural drawings: very high line density, many parallel lines, few circles
        if line_density > 1.0 and line_count > 100 and circle_count < 3:
            return "architectural"
        
        # Mathematical diagrams: moderate lines, some circles, text-like patterns
        if 0.2 < line_density < 0.8 and circle_count > 2 and contour_count < 50:
            return "mathematical"
        
        # Technical drawings: high precision, many small details
        if line_density > 0.8 and contour_count > 50:
            return "technical"
        
        # Default to general
        return "general"
    
    def _get_type_specific_params(self, image_type: str) -> Dict[str, Any]:
        """Get type-specific parameters for detection and processing."""
        params = {
            "circuit": {
                "hough_line_threshold": 80,
                "hough_line_min_length": 40,
                "hough_circle_threshold": 30,
                "canny_low": 50,
                "canny_high": 150,
                "preprocessing_strength": 0.7,
                "expected_primitives": ["line", "circle", "contour"]
            },
            "architectural": {
                "hough_line_threshold": 100,
                "hough_line_min_length": 50,
                "hough_circle_threshold": 50,
                "canny_low": 30,
                "canny_high": 100,
                "preprocessing_strength": 0.8,
                "expected_primitives": ["line", "contour"]
            },
            "mathematical": {
                "hough_line_threshold": 70,
                "hough_line_min_length": 30,
                "hough_circle_threshold": 25,
                "canny_low": 40,
                "canny_high": 120,
                "preprocessing_strength": 0.6,
                "expected_primitives": ["line", "circle", "contour"]
            },
            "technical": {
                "hough_line_threshold": 90,
                "hough_line_min_length": 35,
                "hough_circle_threshold": 35,
                "canny_low": 45,
                "canny_high": 130,
                "preprocessing_strength": 0.75,
                "expected_primitives": ["line", "circle", "contour"]
            },
            "general": {
                "hough_line_threshold": 100,
                "hough_line_min_length": 50,
                "hough_circle_threshold": 30,
                "canny_low": 50,
                "canny_high": 150,
                "preprocessing_strength": 0.7,
                "expected_primitives": ["line", "circle", "contour"]
            }
        }
        
        return params.get(image_type, params["general"])
    
    # ==================== Auto Parameter Tuning ====================
    
    def _auto_tune_params(self, image: np.ndarray, image_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Automatically tune parameters using hybrid strategy.
        
        Strategy: Heuristic first, then adaptive refinement if needed.
        """
        if not self.config.auto_tune_params:
            # Use defaults or config-specified params
            if self.config.opencv_params:
                return self.config.opencv_params
            return self._get_type_specific_params("general")
        
        # Detect image type if not provided
        if image_type is None and self.config.auto_detect_type:
            image_type = self._detect_image_type(image)
        
        # Start with heuristic-based parameters
        params = self._heuristic_tune(image, image_type or "general")
        
        return params
    
    def _heuristic_tune(self, image: np.ndarray, image_type: str) -> Dict[str, Any]:
        """Heuristic-based parameter tuning based on image statistics."""
        # Get base parameters for image type
        params = self._get_type_specific_params(image_type)
        
        # Analyze image to adjust parameters
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        total_pixels = h * w
        
        # Adjust based on image size
        if total_pixels < 100000:  # Small image
            params["hough_line_threshold"] = int(params["hough_line_threshold"] * 0.7)
            params["hough_line_min_length"] = int(params["hough_line_min_length"] * 0.7)
        elif total_pixels > 1000000:  # Large image
            params["hough_line_threshold"] = int(params["hough_line_threshold"] * 1.3)
            params["hough_line_min_length"] = int(params["hough_line_min_length"] * 1.3)
        
        # Adjust based on edge density
        edges = cv2.Canny(gray, params["canny_low"], params["canny_high"])
        edge_density = np.sum(edges > 0) / total_pixels
        
        if edge_density < 0.1:  # Low edge density - lower thresholds
            params["hough_line_threshold"] = int(params["hough_line_threshold"] * 0.8)
            params["canny_low"] = max(20, int(params["canny_low"] * 0.8))
        elif edge_density > 0.3:  # High edge density - raise thresholds
            params["hough_line_threshold"] = int(params["hough_line_threshold"] * 1.2)
            params["canny_high"] = min(255, int(params["canny_high"] * 1.2))
        
        return params
    
    def _adaptive_refine(self, image: np.ndarray, initial_result: AnnotationResult, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptively refine parameters based on initial results."""
        # If no primitives found, relax thresholds
        if len(initial_result.annotation_graph.entities) == 0:
            logger.info("No primitives found, relaxing thresholds")
            params["hough_line_threshold"] = max(30, int(params.get("hough_line_threshold", 100) * 0.7))
            params["hough_circle_threshold"] = max(20, int(params.get("hough_circle_threshold", 30) * 0.7))
            params["canny_low"] = max(20, int(params.get("canny_low", 50) * 0.8))
        
        # If too many primitives, raise thresholds
        elif len(initial_result.annotation_graph.entities) > 500:
            logger.info("Too many primitives, raising thresholds")
            params["hough_line_threshold"] = int(params.get("hough_line_threshold", 100) * 1.3)
            params["hough_circle_threshold"] = int(params.get("hough_circle_threshold", 30) * 1.3)
            params["canny_high"] = min(255, int(params.get("canny_high", 150) * 1.2))
        
        # If low confidence labels, adjust preprocessing
        if initial_result.annotation_graph.labels:
            avg_uncertainty = np.mean([l.uncertainty for l in initial_result.annotation_graph.labels])
            if avg_uncertainty > 0.7:
                logger.info("Low confidence labels, adjusting preprocessing")
                params["preprocessing_strength"] = min(1.0, params.get("preprocessing_strength", 0.7) * 1.2)
        
        return params
    
    # ==================== Retry Logic ====================
    
    def _should_retry(self, result: AnnotationResult, attempt: int) -> bool:
        """Determine if annotation should be retried."""
        if not self.config.auto_retry:
            return False
        
        if attempt >= self.config.max_retries:
            return False
        
        # Retry if no primitives detected
        if len(result.annotation_graph.entities) == 0:
            logger.info(f"Retry {attempt + 1}: No primitives detected")
            return True
        
        # Retry if low confidence
        if result.annotation_graph.labels:
            avg_uncertainty = np.mean([l.uncertainty for l in result.annotation_graph.labels])
            if avg_uncertainty > 0.7:
                logger.info(f"Retry {attempt + 1}: Low confidence (avg uncertainty: {avg_uncertainty:.2f})")
                return True
        
        # Retry if many contradictions
        if len(result.annotation_graph.contradictions) > 5:
            logger.info(f"Retry {attempt + 1}: Many contradictions ({len(result.annotation_graph.contradictions)})")
            return True
        
        return False
    
    def _get_retry_params(self, attempt: int, previous_result: Optional[AnnotationResult], base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for retry attempt."""
        params = base_params.copy()
        
        # Exponential backoff: relax thresholds more with each retry
        backoff_factor = self.config.retry_backoff ** attempt
        
        # Relax thresholds (use defaults if not present)
        params["hough_line_threshold"] = max(30, int(params.get("hough_line_threshold", 100) * (1.0 / backoff_factor)))
        params["hough_circle_threshold"] = max(20, int(params.get("hough_circle_threshold", 30) * (1.0 / backoff_factor)))
        params["canny_low"] = max(20, int(params.get("canny_low", 50) * (1.0 / backoff_factor)))
        
        # Adjust preprocessing
        params["preprocessing_strength"] = min(1.0, params.get("preprocessing_strength", 0.7) * (1.0 + 0.1 * attempt))
        
        return params
    
    def _annotate_with_retry(self, image: np.ndarray, frame_id: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> AnnotationResult:
        """Annotate image with automatic retry logic."""
        if params is None:
            params = self._auto_tune_params(image)
        
        best_result = None
        best_score = -1
        
        for attempt in range(self.config.max_retries):
            try:
                # Use retry parameters if not first attempt
                if attempt > 0:
                    params = self._get_retry_params(attempt, best_result, params)
                    logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retries} with adjusted parameters")
                
                # Perform annotation with current parameters
                result = self._annotate_core(image, frame_id, params)
                
                # Score result (higher is better)
                score = self._score_result(result)
                
                # Keep best result
                if score > best_score:
                    best_result = result
                    best_score = score
                
                # Check if we should retry
                if not self._should_retry(result, attempt):
                    return result
                
                # Wait before retry (exponential backoff)
                if attempt < self.config.max_retries - 1:
                    wait_time = (self.config.retry_backoff ** attempt) * 0.5
                    time.sleep(wait_time)
            
            except Exception as e:
                logger.warning(f"Annotation attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    # Last attempt failed, return best result or error
                    if best_result:
                        return best_result
                    raise
        
        # Return best result after all retries
        return best_result if best_result else AnnotationResult(
            annotation_graph=AnnotationGraph(),
            overlay_image=None,
            formal_report="Error: All retry attempts failed",
            json_output={"error": "All retry attempts failed"},
            processing_time=0.0
        )
    
    def _score_result(self, result: AnnotationResult) -> float:
        """Score annotation result (higher is better)."""
        score = 0.0
        
        # More entities is better
        score += len(result.annotation_graph.entities) * 0.1
        
        # More labels is better
        score += len(result.annotation_graph.labels) * 0.2
        
        # Lower average uncertainty is better
        if result.annotation_graph.labels:
            avg_uncertainty = np.mean([l.uncertainty for l in result.annotation_graph.labels])
            score += (1.0 - avg_uncertainty) * 0.5
        
        # Fewer contradictions is better
        score -= len(result.annotation_graph.contradictions) * 0.3
        
        return max(0.0, score)
    
    # ==================== Smart Caching ====================
    
    def _get_cache_key(self, image: np.ndarray, params: Dict[str, Any]) -> str:
        """Generate cache key from image content and parameters."""
        # Hash image content
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        
        # Hash parameters
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return f"{image_hash}_{params_hash}"
    
    def _get_cached_primitives(self, cache_key: str) -> Optional[List[PrimitiveEntity]]:
        """Get cached primitives if available."""
        if not self.config.cache_enabled:
            return None
        
        cache_file = self._cache_dir / f"{cache_key}_primitives.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                primitives = [PrimitiveEntity(**item) for item in data]
                logger.debug(f"Loaded {len(primitives)} primitives from cache")
                return primitives
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        return None
    
    def _cache_primitives(self, cache_key: str, primitives: List[PrimitiveEntity]) -> None:
        """Cache extracted primitives."""
        if not self.config.cache_enabled:
            return
        
        try:
            cache_file = self._cache_dir / f"{cache_key}_primitives.json"
            data = [p.model_dump() for p in primitives]
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Cached {len(primitives)} primitives")
        except Exception as e:
            logger.warning(f"Error caching primitives: {e}")
    
    # ==================== Image Preprocessing ====================
    
    def _preprocess_image(self, image: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Main preprocessing pipeline to reduce input entropy.
        
        Args:
            image: Input image as numpy array
            params: Optional parameters dict (uses defaults if None)
            
        Returns:
            Preprocessed image
        """
        if params is None:
            params = {}
        
        processed = image.copy()
        
        # Get preprocessing strength
        preprocessing_strength = params.get("preprocessing_strength", 0.7)
        
        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization (scaled by strength)
        if preprocessing_strength > 0.3:
            processed = self._adaptive_histogram_equalization(processed)
        
        # Edge amplification (scaled by strength)
        if preprocessing_strength > 0.5:
            processed = self._edge_amplification(processed, strength=preprocessing_strength)
        
        # Noise-aware downscaling (if image is too large)
        max_dimension = 2048
        h, w = processed.shape[:2]
        if max(h, w) > max_dimension:
            target_size = (int(w * max_dimension / max(h, w)), int(h * max_dimension / max(h, w)))
            processed = self._noise_aware_downscale(processed, target_size)
        
        return processed
    
    def _adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization."""
        if not SKIMAGE_AVAILABLE:
            # Fallback to OpenCV CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        
        # Use skimage for better quality
        return exposure.equalize_adapthist(image, clip_limit=0.03)
    
    def _edge_amplification(self, image: np.ndarray, strength: float = 0.7) -> np.ndarray:
        """Amplify edges using Laplacian operator."""
        # Ensure image is in correct format (uint8)
        if image.dtype != np.uint8:
            # Normalize to 0-255 range if float
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.absolute(laplacian)
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
        
        # Combine with original (strength controls amplification)
        edge_weight = min(0.5, strength * 0.5)
        original_weight = 1.0 - edge_weight
        amplified = cv2.addWeighted(image, original_weight, laplacian, edge_weight, 0)
        return amplified
    
    def _noise_aware_downscale(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Downscale image using wavelets to preserve important features."""
        if not PYWAVELETS_AVAILABLE:
            # Fallback to standard resize
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Use wavelets for better downscaling
        # Simple approach: resize with area interpolation (wavelets would be more complex)
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # ==================== Vision Primitives Extraction ====================
    
    def _extract_lines(self, image: np.ndarray, params: Optional[Dict[str, Any]] = None) -> List[Line]:
        """
        Extract lines using Hough line detection.
        
        Args:
            image: Preprocessed image
            params: Optional parameters dict (uses defaults if None)
            
        Returns:
            List of Line objects with pixel coordinates
        """
        if params is None:
            params = {}
        
        lines = []
        
        # Get parameters with defaults
        canny_low = params.get("canny_low", 50)
        canny_high = params.get("canny_high", 150)
        hough_threshold = params.get("hough_line_threshold", 100)
        min_line_length = params.get("hough_line_min_length", 50)
        max_line_gap = params.get("hough_line_max_gap", 10)
        
        # Edge detection
        edges = cv2.Canny(image, canny_low, canny_high, apertureSize=3)
        
        # Hough line detection
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                lines.append(Line(
                    start_point=(int(x1), int(y1)),
                    end_point=(int(x2), int(y2))
                ))
        
        logger.debug(f"Extracted {len(lines)} lines")
        return lines
    
    def _extract_circles(self, image: np.ndarray, params: Optional[Dict[str, Any]] = None) -> List[Circle]:
        """
        Extract circles using Hough circle detection.
        
        Args:
            image: Preprocessed image
            params: Optional parameters dict (uses defaults if None)
            
        Returns:
            List of Circle objects with center and radius
        """
        if params is None:
            params = {}
        
        circles = []
        
        # Get parameters with defaults
        hough_threshold = params.get("hough_circle_threshold", 30)
        min_dist = params.get("hough_circle_min_dist", 30)
        min_radius = params.get("hough_circle_min_radius", 10)
        max_radius = params.get("hough_circle_max_radius", 0)  # 0 means no maximum
        
        h, w = image.shape[:2]
        if max_radius == 0:
            max_radius = min(h, w) // 2
        
        # Hough circle detection
        detected_circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=50,
            param2=hough_threshold,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for circle in detected_circles[0, :]:
                center_x, center_y, radius = circle
                circles.append(Circle(
                    center=(int(center_x), int(center_y)),
                    radius=float(radius)
                ))
        
        logger.debug(f"Extracted {len(circles)} circles")
        return circles
    
    def _extract_contours(self, image: np.ndarray, params: Optional[Dict[str, Any]] = None) -> List[Contour]:
        """
        Extract contours from image.
        
        Args:
            image: Preprocessed image
            params: Optional parameters dict (uses defaults if None)
            
        Returns:
            List of Contour objects
        """
        if params is None:
            params = {}
        
        contours_list = []
        
        # Get parameters with defaults
        canny_low = params.get("canny_low", 50)
        canny_high = params.get("canny_high", 150)
        
        # Edge detection
        edges = cv2.Canny(image, canny_low, canny_high)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) >= 3:  # Need at least 3 points for a polygon
                points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                contours_list.append(Contour(points=points))
        
        logger.debug(f"Extracted {len(contours_list)} contours")
        return contours_list
    
    def _compute_intersections(self, primitives: List[PrimitiveEntity]) -> List[Intersection]:
        """
        Compute intersections between primitives.
        
        Args:
            primitives: List of primitive entities
            
        Returns:
            List of Intersection objects
        """
        intersections = []
        
        # Group primitives by type
        lines = [p for p in primitives if p.primitive_type == "line"]
        circles = [p for p in primitives if p.primitive_type == "circle"]
        
        # Line-line intersections
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], start=i+1):
                intersection_point = self._line_line_intersection(
                    line1.pixel_coords[0], line1.pixel_coords[1],
                    line2.pixel_coords[0], line2.pixel_coords[1]
                )
                if intersection_point:
                    intersections.append(Intersection(
                        point=intersection_point,
                        primitive_ids=[line1.id, line2.id]
                    ))
        
        # Line-circle intersections
        for line in lines:
            for circle in circles:
                if len(line.pixel_coords) >= 2 and len(circle.pixel_coords) >= 1:
                    intersection_points = self._line_circle_intersection(
                        line.pixel_coords[0], line.pixel_coords[1],
                        circle.pixel_coords[0], circle.metadata.get("radius", 0)
                    )
                    for point in intersection_points:
                        intersections.append(Intersection(
                            point=point,
                            primitive_ids=[line.id, circle.id]
                        ))
        
        logger.debug(f"Computed {len(intersections)} intersections")
        return intersections
    
    def _line_line_intersection(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        p3: Tuple[int, int],
        p4: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Compute intersection of two lines."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            return (x, y)
        
        return None
    
    def _line_circle_intersection(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        center: Tuple[int, int],
        radius: float
    ) -> List[Tuple[int, int]]:
        """Compute intersection of line and circle."""
        if radius <= 0:
            return []
        
        # Convert to float for computation
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        cx, cy = float(center[0]), float(center[1])
        
        # Line equation: ax + by + c = 0
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return []
        
        # Distance from center to line
        a = dy
        b = -dx
        c = dx * y1 - dy * x1
        
        dist = abs(a * cx + b * cy + c) / np.sqrt(a * a + b * b)
        
        if dist > radius:
            return []
        
        # Compute intersection points (simplified)
        # This is a simplified version - full implementation would solve quadratic
        intersections = []
        
        # Project center onto line
        if abs(dx) > abs(dy):
            t = (cx - x1) / dx if abs(dx) > 1e-10 else 0
        else:
            t = (cy - y1) / dy if abs(dy) > 1e-10 else 0
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # Check if projection is on line segment
        if 0 <= t <= 1:
            # Distance from projection to intersection
            h = np.sqrt(radius * radius - dist * dist)
            
            # Direction vector
            length = np.sqrt(dx * dx + dy * dy)
            if length > 1e-10:
                dir_x = dx / length
                dir_y = dy / length
                
                # Two intersection points
                for sign in [-1, 1]:
                    ix = int(proj_x + sign * h * dir_x)
                    iy = int(proj_y + sign * h * dir_y)
                    intersections.append((ix, iy))
        
        return intersections
    
    def _validate_primitive(self, primitive: PrimitiveEntity, image: np.ndarray) -> bool:
        """Verify primitive exists in image."""
        if not primitive.pixel_coords:
            return False
        
        h, w = image.shape[:2]
        for x, y in primitive.pixel_coords:
            if not (0 <= x < w and 0 <= y < h):
                return False
        
        return True
    
    # ==================== GPT-4o-mini Labeling ====================
    
    def _label_primitives(
        self,
        primitives: List[PrimitiveEntity],
        image: np.ndarray
    ) -> List[SemanticLabel]:
        """
        Label primitives using restricted GPT-4o-mini.
        
        Args:
            primitives: List of primitive entities to label
            image: Original image (for context)
            
        Returns:
            List of semantic labels
        """
        if not primitives:
            return []
        
        # Prepare prompt with primitive information
        primitive_info = []
        for prim in primitives:
            prim_type = prim.primitive_type
            coords = prim.pixel_coords
            if prim_type == "line" and len(coords) >= 2:
                info = f"Line {prim.id}: from {coords[0]} to {coords[1]}"
            elif prim_type == "circle" and len(coords) >= 1:
                radius = prim.metadata.get("radius", "unknown")
                info = f"Circle {prim.id}: center {coords[0]}, radius {radius}"
            elif prim_type == "contour":
                info = f"Contour {prim.id}: {len(coords)} points"
            else:
                info = f"{prim_type} {prim.id}: {len(coords)} coordinates"
            primitive_info.append(info)
        
        prompt = f"""Label the following geometric primitives extracted from an image.

Primitives:
{chr(10).join(primitive_info)}

For each primitive, provide:
- entity_id: The ID from the list above
- label: A semantic description (e.g., "resistor", "wire", "boundary", "component")
- uncertainty: A number between 0.0 (certain) and 1.0 (uncertain)
- tentative: false for labels
- reasoning: Brief explanation (optional)

Return your response as a JSON array of labels, where each label has:
{{"entity_id": "...", "label": "...", "uncertainty": 0.0-1.0, "tentative": false, "reasoning": "..."}}

Only label primitives that exist in the provided list. Do not invent new primitives."""
        
        try:
            # Convert image to base64 for vision model
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            img_data_url = f"data:image/jpeg;base64,{img_base64}"
            
            # Call GPT-4o-mini
            response = self._labeler.run(task=prompt, img=img_data_url)
            
            # Parse response
            labels = self._parse_label_response(response, primitives)
            
            # Validate labels (only accept primitives that exist)
            valid_labels = []
            primitive_ids = {p.id for p in primitives}
            for label in labels:
                if label.entity_id in primitive_ids:
                    valid_labels.append(label)
                else:
                    logger.warning(f"Label references non-existent entity: {label.entity_id}")
            
            logger.info(f"Labeled {len(valid_labels)}/{len(primitives)} primitives")
            return valid_labels
            
        except Exception as e:
            logger.error(f"Error labeling primitives: {e}")
            return []
    
    def _parse_label_response(
        self,
        response: Union[str, Dict, Any],
        primitives: List[PrimitiveEntity]
    ) -> List[SemanticLabel]:
        """Parse GPT response into SemanticLabel objects."""
        labels = []
        
        # Extract JSON from response
        response_str = str(response)
        
        # Try to find JSON array in response
        import re
        json_match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if json_match:
            try:
                label_data = json.loads(json_match.group())
                for item in label_data:
                    try:
                        label = SemanticLabel(
                            entity_id=item.get("entity_id", ""),
                            label=item.get("label", "unknown"),
                            uncertainty=float(item.get("uncertainty", 0.5)),
                            tentative=bool(item.get("tentative", False)),
                            reasoning=item.get("reasoning")
                        )
                        labels.append(label)
                    except Exception as e:
                        logger.warning(f"Failed to parse label: {e}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from response: {e}")
        
        return labels
    
    def _suggest_relations(self, entities: List[PrimitiveEntity]) -> List[Relation]:
        """
        Suggest relations between entities (flagged as tentative).
        
        Args:
            entities: List of entities
            
        Returns:
            List of tentative relations
        """
        # This would use GPT-4o-mini to suggest relations
        # For now, return empty list (can be implemented later)
        return []
    
    # ==================== Graph Compilation ====================
    
    def _compile_graph(self, annotations: AnnotationGraph) -> rx.PyDiGraph:
        """
        Compile annotations into typed graph using rustworkx.
        
        Args:
            annotations: Annotation graph
            
        Returns:
            rustworkx directed graph
        """
        graph = rx.PyDiGraph()
        
        # Add nodes (entities)
        node_map = {}  # entity_id -> node index
        for entity in annotations.entities:
            node_idx = graph.add_node(entity)
            node_map[entity.id] = node_idx
        
        # Add edges (relations)
        for relation in annotations.relations:
            if relation.source_id in node_map and relation.target_id in node_map:
                source_idx = node_map[relation.source_id]
                target_idx = node_map[relation.target_id]
                graph.add_edge(source_idx, target_idx, relation)
        
        return graph
    
    def _detect_cycles(self, graph: rx.PyDiGraph) -> List[List[str]]:
        """Detect cycles in the graph."""
        cycles = []
        
        # Simple cycle detection using DFS
        # This is a simplified version - rustworkx has cycle detection methods
        try:
            # Check if graph is acyclic
            if not rx.is_directed_acyclic_graph(graph):
                # Find cycles (simplified - would need proper cycle enumeration)
                cycles.append(["cycle_detected"])
        except Exception as e:
            logger.warning(f"Cycle detection error: {e}")
        
        return cycles
    
    def _detect_mutually_exclusive(self, graph: rx.PyDiGraph) -> List[Tuple[str, str]]:
        """Detect mutually exclusive relations."""
        # This would check for conflicting relation types
        # For now, return empty list
        return []
    
    def _detect_unsupported_relations(
        self,
        graph: rx.PyDiGraph,
        primitives: List[PrimitiveEntity]
    ) -> List[str]:
        """Detect relations that reference non-existent primitives."""
        unsupported = []
        primitive_ids = {p.id for p in primitives}
        
        # Check all edges
        for edge in graph.edge_list():
            source_idx, target_idx = edge
            source_node = graph[source_idx]
            target_node = graph[target_idx]
            
            if isinstance(source_node, PrimitiveEntity) and isinstance(target_node, PrimitiveEntity):
                if source_node.id not in primitive_ids or target_node.id not in primitive_ids:
                    unsupported.append(f"relation_{source_node.id}_{target_node.id}")
        
        return unsupported
    
    def _validate_graph(self, graph: rx.PyDiGraph) -> Dict[str, Any]:
        """Comprehensive graph validation."""
        validation_result = {
            "is_valid": True,
            "cycles": [],
            "mutually_exclusive": [],
            "unsupported_relations": [],
            "warnings": []
        }
        
        # Check for cycles
        cycles = self._detect_cycles(graph)
        if cycles:
            validation_result["is_valid"] = False
            validation_result["cycles"] = cycles
        
        # Check for mutually exclusive relations
        mutually_exclusive = self._detect_mutually_exclusive(graph)
        if mutually_exclusive:
            validation_result["mutually_exclusive"] = mutually_exclusive
        
        return validation_result
    
    # ==================== Deterministic Math Layer ====================
    
    def _compute_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _compute_angle(self, line1: Line, line2: Line) -> float:
        """Compute angle between two lines in degrees."""
        # Get direction vectors
        dx1 = line1.end_point[0] - line1.start_point[0]
        dy1 = line1.end_point[1] - line1.start_point[1]
        dx2 = line2.end_point[0] - line2.start_point[0]
        dy2 = line2.end_point[1] - line2.start_point[1]
        
        # Compute angle using dot product
        dot = dx1 * dx2 + dy1 * dy2
        mag1 = np.sqrt(dx1**2 + dy1**2)
        mag2 = np.sqrt(dx2**2 + dy2**2)
        
        if mag1 < 1e-10 or mag2 < 1e-10:
            return 0.0
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _solve_constraints(self, constraints: List[Any]) -> Dict[str, float]:
        """Solve constraints using z3-solver."""
        if not Z3_AVAILABLE:
            logger.warning("z3-solver not available, constraint solving disabled")
            return {}
        
        # This would use z3 to solve constraints
        # Simplified implementation
        return {}
    
    def _verify_geometry(self, primitives: List[PrimitiveEntity]) -> bool:
        """Check geometric consistency."""
        # Basic validation: check that coordinates are valid
        for prim in primitives:
            if not self._validate_primitive(prim, np.zeros((100, 100), dtype=np.uint8)):
                return False
        return True
    
    # ==================== CR-CA Integration ====================
    
    def _create_claim(
        self,
        annotation: SemanticLabel,
        dependencies: List[str]
    ) -> Claim:
        """Create a claim for failure-aware reasoning."""
        claim = Claim(
            annotation=annotation,
            dependencies=dependencies,
            robustness_score=1.0
        )
        self._claims[claim.claim_id] = claim
        return claim
    
    def _trace_dependencies(self, claim_id: str) -> List[str]:
        """Trace dependency chain for a claim."""
        if claim_id not in self._claims:
            return []
        
        visited = set()
        dependencies = []
        
        def _collect_deps(cid: str):
            if cid in visited or cid not in self._claims:
                return
            visited.add(cid)
            claim = self._claims[cid]
            for dep_id in claim.dependencies:
                if dep_id not in visited:
                    _collect_deps(dep_id)
                    dependencies.append(dep_id)
        
        _collect_deps(claim_id)
        return dependencies
    
    def _counterfactual_explanation(
        self,
        claim_id: str,
        removed_dependency: str
    ) -> str:
        """Generate counterfactual explanation."""
        if claim_id not in self._claims:
            return "Claim not found"
        
        claim = self._claims[claim_id]
        if removed_dependency not in claim.dependencies:
            return "Dependency not found in claim"
        
        return f"If dependency {removed_dependency} were removed, claim {claim_id} would be invalidated."
    
    def _robustness_analysis(self, claims: List[Claim]) -> Dict[str, float]:
        """Analyze robustness of claims."""
        robustness_scores = {}
        
        for claim in claims:
            # Simple robustness: inverse of uncertainty and dependency count
            uncertainty_penalty = claim.annotation.uncertainty
            dependency_penalty = len(claim.dependencies) * 0.1
            
            robustness = max(0.0, 1.0 - uncertainty_penalty - dependency_penalty)
            robustness_scores[claim.claim_id] = robustness
            claim.robustness_score = robustness
        
        return robustness_scores
    
    # ==================== Temporal Coherence ====================
    
    def _track_entity(
        self,
        entity: PrimitiveEntity,
        frame_id: int
    ) -> str:
        """Track entity across frames using Kalman filter."""
        if not self.enable_temporal_tracking:
            return entity.id
        
        # Get or create Kalman filter for entity
        if entity.id not in self._entity_trackers:
            if FILTERPY_AVAILABLE:
                kf = KalmanFilter(dim_x=4, dim_z=2)  # x, y, vx, vy
                # Initialize with entity position
                if entity.pixel_coords:
                    center = self._get_entity_center(entity)
                    kf.x = np.array([center[0], center[1], 0.0, 0.0])
                    kf.P *= 1000.0  # Initial uncertainty
                    kf.R = np.eye(2) * 5  # Measurement noise
                    kf.Q = np.eye(4) * 0.1  # Process noise
                self._entity_trackers[entity.id] = kf
            else:
                # Fallback: simple ID mapping
                self._entity_trackers[entity.id] = {"center": self._get_entity_center(entity)}
        
        # Update tracker
        if FILTERPY_AVAILABLE and entity.id in self._entity_trackers:
            kf = self._entity_trackers[entity.id]
            if isinstance(kf, KalmanFilter) and entity.pixel_coords:
                center = self._get_entity_center(entity)
                kf.predict()
                kf.update(np.array([center[0], center[1]]))
        
        return entity.id
    
    def _get_entity_center(self, entity: PrimitiveEntity) -> Tuple[float, float]:
        """Get center point of entity."""
        if not entity.pixel_coords:
            return (0.0, 0.0)
        
        if entity.primitive_type == "circle" and len(entity.pixel_coords) >= 1:
            return (float(entity.pixel_coords[0][0]), float(entity.pixel_coords[0][1]))
        elif entity.primitive_type == "line" and len(entity.pixel_coords) >= 2:
            x1, y1 = entity.pixel_coords[0]
            x2, y2 = entity.pixel_coords[1]
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        else:
            # Use centroid of all points
            xs = [p[0] for p in entity.pixel_coords]
            ys = [p[1] for p in entity.pixel_coords]
            return (float(np.mean(xs)), float(np.mean(ys)))
    
    def _predict_next_position(self, entity_id: str) -> Tuple[float, float]:
        """Predict next position using Kalman filter."""
        if not self.enable_temporal_tracking or entity_id not in self._entity_trackers:
            return (0.0, 0.0)
        
        tracker = self._entity_trackers[entity_id]
        if FILTERPY_AVAILABLE and isinstance(tracker, KalmanFilter):
            tracker.predict()
            return (float(tracker.x[0]), float(tracker.x[1]))
        elif isinstance(tracker, dict) and "center" in tracker:
            return tracker["center"]
        
        return (0.0, 0.0)
    
    def _check_continuity(
        self,
        entity_id: str,
        new_annotation: SemanticLabel
    ) -> bool:
        """Verify consistency of annotation across frames."""
        # Check if label changed significantly
        if entity_id in self._claims:
            old_claim = self._claims[entity_id]
            old_label = old_claim.annotation.label
            new_label = new_annotation.label
            
            # Simple check: if labels are very different, flag as inconsistent
            if old_label.lower() != new_label.lower():
                return False
        
        return True
    
    def _detect_instability(self, entity_id: str) -> Dict[str, Any]:
        """Detect temporal instability (angle jumps, position shifts)."""
        instability = {
            "detected": False,
            "reason": None,
            "metrics": {}
        }
        
        if not self.enable_temporal_tracking or entity_id not in self._entity_trackers:
            return instability
        
        # Check position variance
        if len(self._frame_history) >= 2:
            # Compare positions across frames
            positions = []
            for graph in self._frame_history[-5:]:  # Last 5 frames
                entity = graph.get_entity_by_id(entity_id)
                if entity:
                    center = self._get_entity_center(entity)
                    positions.append(center)
            
            if len(positions) >= 2:
                # Compute position variance
                positions_array = np.array(positions)
                variance = np.var(positions_array, axis=0)
                max_variance = np.max(variance)
                
                if max_variance > 100.0:  # Threshold
                    instability["detected"] = True
                    instability["reason"] = f"High position variance: {max_variance:.2f}"
                    instability["metrics"]["position_variance"] = float(max_variance)
        
        return instability
    
    # ==================== Output Generation ====================
    
    def _generate_overlay(
        self,
        image: np.ndarray,
        annotations: AnnotationGraph
    ) -> np.ndarray:
        """Generate overlay image with annotations drawn."""
        overlay = image.copy()
        
        # Draw entities
        for entity in annotations.entities:
            if entity.primitive_type == "line" and len(entity.pixel_coords) >= 2:
                cv2.line(
                    overlay,
                    entity.pixel_coords[0],
                    entity.pixel_coords[1],
                    (0, 255, 0),
                    2
                )
            elif entity.primitive_type == "circle" and len(entity.pixel_coords) >= 1:
                center = entity.pixel_coords[0]
                radius = int(entity.metadata.get("radius", 10))
                cv2.circle(overlay, center, radius, (255, 0, 0), 2)
            elif entity.primitive_type == "contour":
                points = np.array(entity.pixel_coords, dtype=np.int32)
                cv2.polylines(overlay, [points], True, (0, 0, 255), 2)
        
        # Draw labels
        for label in annotations.labels:
            entity = annotations.get_entity_by_id(label.entity_id)
            if entity and entity.pixel_coords:
                center = self._get_entity_center(entity)
                center_int = (int(center[0]), int(center[1]))
                cv2.putText(
                    overlay,
                    label.label,
                    center_int,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return overlay
    
    def _generate_formal_report(
        self,
        annotations: AnnotationGraph,
        contradictions: List[Contradiction]
    ) -> str:
        """Generate structured formal report."""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("FORMAL ANNOTATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # KNOWN section
        report_lines.append("KNOWN:")
        report_lines.append("-" * 80)
        for entity in annotations.entities:
            coords_str = ", ".join([f"({x},{y})" for x, y in entity.pixel_coords[:5]])
            if len(entity.pixel_coords) > 5:
                coords_str += f" ... ({len(entity.pixel_coords)} total points)"
            report_lines.append(f"  {entity.primitive_type.upper()} {entity.id}: {coords_str}")
        
        # Measured quantities
        report_lines.append("")
        report_lines.append("  Measured quantities:")
        for i, entity1 in enumerate(annotations.entities):
            for entity2 in annotations.entities[i+1:]:
                if len(entity1.pixel_coords) >= 1 and len(entity2.pixel_coords) >= 1:
                    center1 = self._get_entity_center(entity1)
                    center2 = self._get_entity_center(entity2)
                    distance = self._compute_distance(center1, center2)
                    report_lines.append(f"    Distance {entity1.id} <-> {entity2.id}: {distance:.2f} pixels")
        
        # ASSUMED section
        report_lines.append("")
        report_lines.append("ASSUMED:")
        report_lines.append("-" * 80)
        tentative_labels = [l for l in annotations.labels if l.tentative]
        tentative_relations = [r for r in annotations.relations if r.tentative]
        
        for label in tentative_labels:
            report_lines.append(f"  Label '{label.label}' for entity {label.entity_id} (uncertainty: {label.uncertainty:.2f})")
        
        for relation in tentative_relations:
            report_lines.append(f"  Relation {relation.relation_type} between {relation.source_id} and {relation.target_id}")
        
        # AMBIGUOUS section
        report_lines.append("")
        report_lines.append("AMBIGUOUS:")
        report_lines.append("-" * 80)
        uncertain_labels = [l for l in annotations.labels if l.uncertainty > 0.5]
        for label in uncertain_labels:
            report_lines.append(f"  {label.entity_id}: {label.label} (uncertainty: {label.uncertainty:.2f})")
        
        # CONTRADICTS section
        if contradictions:
            report_lines.append("")
            report_lines.append("CONTRADICTS:")
            report_lines.append("-" * 80)
            for contradiction in contradictions:
                report_lines.append(f"  {contradiction.contradiction_type}: {contradiction.description}")
                report_lines.append(f"    Entities involved: {', '.join(contradiction.entities_involved)}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _generate_json(self, annotations: AnnotationGraph) -> Dict[str, Any]:
        """Generate JSON representation of annotations."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "type": e.primitive_type,
                    "pixel_coords": e.pixel_coords,
                    "metadata": e.metadata
                }
                for e in annotations.entities
            ],
            "labels": [
                {
                    "entity_id": l.entity_id,
                    "label": l.label,
                    "uncertainty": l.uncertainty,
                    "tentative": l.tentative,
                    "reasoning": l.reasoning
                }
                for l in annotations.labels
            ],
            "relations": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation_type": r.relation_type,
                    "uncertainty": r.uncertainty,
                    "tentative": r.tentative
                }
                for r in annotations.relations
            ],
            "contradictions": [
                {
                    "type": c.contradiction_type,
                    "entities_involved": c.entities_involved,
                    "description": c.description,
                    "severity": c.severity
                }
                for c in annotations.contradictions
            ]
        }
    
    # ==================== Core Annotation Logic ====================
    
    def _annotate_core(
        self,
        image: np.ndarray,
        frame_id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> AnnotationResult:
        """
        Core annotation logic (internal method).
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            frame_id: Optional frame ID for temporal tracking
            params: Optional parameters dict for tuning
            
        Returns:
            AnnotationResult with overlay, report, and JSON
        """
        start_time = time.time()
        
        try:
            # Get or generate cache key
            if params is None:
                params = self._auto_tune_params(image)
            cache_key = self._get_cache_key(image, params) if self.config.cache_enabled else None
            
            # Try to load cached primitives
            cached_primitives = None
            if cache_key:
                cached_primitives = self._get_cached_primitives(cache_key)
            
            # 1. Preprocess image
            processed_image = self._preprocess_image(image, params)
            
            # 2. Extract primitives (use cache if available)
            if cached_primitives:
                entities = cached_primitives
                logger.debug(f"Using {len(entities)} cached primitives")
            else:
                lines = self._extract_lines(processed_image, params)
                circles = self._extract_circles(processed_image, params)
                contours = self._extract_contours(processed_image, params)
                
                # Convert to PrimitiveEntity objects
                entities = []
                for line in lines:
                    entity = PrimitiveEntity(
                        id=line.entity_id or str(len(entities)),
                        pixel_coords=[line.start_point, line.end_point],
                        primitive_type="line"
                    )
                    entities.append(entity)
                    line.entity_id = entity.id
                
                for circle in circles:
                    entity = PrimitiveEntity(
                        id=circle.entity_id or str(len(entities)),
                        pixel_coords=[circle.center],
                        primitive_type="circle",
                        metadata={"radius": circle.radius}
                    )
                    entities.append(entity)
                    circle.entity_id = entity.id
                
                for contour in contours:
                    entity = PrimitiveEntity(
                        id=contour.entity_id or str(len(entities)),
                        pixel_coords=contour.points,
                        primitive_type="contour"
                    )
                    entities.append(entity)
                    contour.entity_id = entity.id
                
                # Compute intersections
                intersections = self._compute_intersections(entities)
                for intersection in intersections:
                    entity = PrimitiveEntity(
                        id=intersection.entity_id or str(len(entities)),
                        pixel_coords=[intersection.point],
                        primitive_type="intersection",
                        metadata={"primitive_ids": intersection.primitive_ids}
                    )
                    entities.append(entity)
                    intersection.entity_id = entity.id
                
                # Cache primitives
                if cache_key:
                    self._cache_primitives(cache_key, entities)
            
            # 3. Label primitives (GPT-4o-mini, restricted) - never cached
            labels = self._label_primitives(entities, image)
            
            # 4. Compile graph
            annotation_graph = AnnotationGraph(
                entities=entities,
                labels=labels,
                relations=[],
                contradictions=[],
                metadata={"image_shape": image.shape, "image_type": params.get("detected_type", "unknown")}
            )
            
            graph = self._compile_graph(annotation_graph)
            
            # 5. Detect contradictions
            validation = self._validate_graph(graph)
            contradictions = []
            if validation.get("cycles"):
                contradictions.append(Contradiction(
                    contradiction_type="cycle",
                    entities_involved=[],
                    description="Cyclic dependencies detected in graph",
                    severity="high"
                ))
            if validation.get("unsupported_relations"):
                contradictions.append(Contradiction(
                    contradiction_type="unsupported",
                    entities_involved=[],
                    description=f"Unsupported relations: {validation['unsupported_relations']}",
                    severity="medium"
                ))
            
            annotation_graph.contradictions = contradictions
            
            # 6. Compute measurements (deterministic math)
            # Already done in _generate_formal_report
            
            # 7. Track temporally (Kalman filters)
            instability_detected = False
            instability_reason = None
            if self.enable_temporal_tracking and frame_id is not None:
                for entity in entities:
                    self._track_entity(entity, frame_id)
                    instability = self._detect_instability(entity.id)
                    if instability["detected"]:
                        instability_detected = True
                        instability_reason = instability["reason"]
                
                self._frame_history.append(annotation_graph)
                if len(self._frame_history) > 10:  # Keep last 10 frames
                    self._frame_history.pop(0)
            
            # 8. Generate outputs
            overlay_image = self._generate_overlay(image, annotation_graph)
            formal_report = self._generate_formal_report(annotation_graph, contradictions)
            json_output = self._generate_json(annotation_graph)
            
            # Serialize overlay image
            _, buffer = cv2.imencode('.jpg', overlay_image)
            overlay_bytes = buffer.tobytes()
            
            processing_time = time.time() - start_time
            
            result = AnnotationResult(
                annotation_graph=annotation_graph,
                overlay_image=overlay_bytes,
                formal_report=formal_report,
                json_output=json_output,
                instability_detected=instability_detected,
                instability_reason=instability_reason,
                processing_time=processing_time,
                frame_id=frame_id
            )
            
            logger.info(f"Annotation complete: {len(entities)} entities, {len(labels)} labels, {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in annotation: {e}", exc_info=True)
            # Return empty result on error
            return AnnotationResult(
                annotation_graph=AnnotationGraph(),
                overlay_image=None,
                formal_report=f"Error: {str(e)}",
                json_output={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    # ==================== Main Public API ====================
    
    def annotate(
        self,
        input: Union[str, np.ndarray, Image.Image, List, Path],
        frame_id: Optional[int] = None,
        output: Optional[str] = None
    ) -> Union[AnnotationResult, np.ndarray, Dict[str, Any], str, List]:
        """
        Annotate image(s) with full automation.
        
        Automatically handles:
        - Input type detection (file path, URL, numpy array, PIL Image, batch)
        - Image type detection (circuit, architectural, mathematical, etc.)
        - Parameter tuning
        - Retry logic
        - Caching
        - Batch processing with parallelization
        
        Args:
            input: Image input - can be:
                - File path (str or Path): "image.png"
                - URL (str): "https://example.com/image.png"
                - NumPy array: np.ndarray
                - PIL Image: PIL.Image.Image
                - List: [path1, path2, ...] for batch processing
            frame_id: Optional frame ID for temporal tracking
            output: Output format - "overlay" (default), "json", "report", "all"
                    If None, uses config.output_format
            
        Returns:
            - If output="overlay" (default): numpy array (overlay image)
            - If output="json": dict (JSON data)
            - If output="report": str (formal report)
            - If output="all": AnnotationResult object
            - If input is list: List of above (based on output format)
        """
        # Determine output format
        # For backward compatibility: if input is np.ndarray and output is None, return AnnotationResult
        is_old_api = isinstance(input, np.ndarray) and output is None
        
        if is_old_api:
            output_format = "all"  # Return full AnnotationResult for backward compatibility
        else:
            output_format = output or self.config.output_format
        
        # Auto-detect if input is batch
        input_type = self._detect_input_type(input)
        if input_type == 'batch':
            return self._annotate_batch_auto(input, frame_id, output_format)
        
        # Single image processing
        # Auto-load input
        try:
            image = self._auto_load_input(input)
        except Exception as e:
            logger.error(f"Error loading input: {e}")
            if output_format == "overlay":
                return np.zeros((100, 100, 3), dtype=np.uint8)  # Empty image
            elif output_format == "json":
                return {"error": str(e)}
            elif output_format == "report":
                return f"Error: {str(e)}"
            else:
                return AnnotationResult(
                    annotation_graph=AnnotationGraph(),
                    overlay_image=None,
                    formal_report=f"Error: {str(e)}",
                    json_output={"error": str(e)},
                    processing_time=0.0
                )
        
        # Auto-detect image type and tune parameters
        image_type = None
        if self.config.auto_detect_type:
            image_type = self._detect_image_type(image)
            logger.info(f"Detected image type: {image_type}")
        
        # Auto-tune parameters
        params = self._auto_tune_params(image, image_type)
        if image_type:
            params["detected_type"] = image_type
        
        # Perform annotation with retry logic
        result = self._annotate_with_retry(image, frame_id, params)
        
        # Return in requested format
        return self._format_output(result, output_format)
    
    def _format_output(self, result: AnnotationResult, output_format: str) -> Union[AnnotationResult, np.ndarray, Dict[str, Any], str]:
        """Format annotation result according to output format."""
        if output_format == "overlay":
            # Return numpy array
            if result.overlay_image:
                return cv2.imdecode(
                    np.frombuffer(result.overlay_image, np.uint8),
                    cv2.IMREAD_COLOR
                )
            else:
                return np.zeros((100, 100, 3), dtype=np.uint8)
        
        elif output_format == "json":
            return result.json_output
        
        elif output_format == "report":
            return result.formal_report
        
        else:  # "all" or default
            return result
    
    def _annotate_batch_auto(
        self,
        inputs: List,
        frame_id: Optional[int] = None,
        output_format: str = "overlay"
    ) -> List:
        """
        Automatically batch process multiple inputs.
        
        Features:
        - Auto-detects input types
        - Parallel processing
        - Progress reporting (tqdm)
        - Error recovery (continues on individual failures)
        - Smart caching
        """
        # Determine number of workers
        if self.config.parallel_workers is None:
            import os
            workers = min(len(inputs), os.cpu_count() or 4)
        else:
            workers = self.config.parallel_workers
        
        results = []
        
        # Use progress bar if available
        if TQDM_AVAILABLE and self.config.show_progress:
            iterator = tqdm(inputs, desc="Annotating images")
        else:
            iterator = inputs
        
        # Process in parallel if multiple workers
        if workers > 1 and len(inputs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_input = {
                    executor.submit(self._annotate_single_with_error_handling, inp, i, output_format): (i, inp)
                    for i, inp in enumerate(inputs)
                }
                
                # Collect results in order
                results_dict = {}
                for future in as_completed(future_to_input):
                    idx, inp = future_to_input[future]
                    try:
                        result = future.result()
                        results_dict[idx] = result
                    except Exception as e:
                        logger.error(f"Error processing input {idx}: {e}")
                        # Return empty result based on output format
                        if output_format == "overlay":
                            results_dict[idx] = np.zeros((100, 100, 3), dtype=np.uint8)
                        elif output_format == "json":
                            results_dict[idx] = {"error": str(e)}
                        elif output_format == "report":
                            results_dict[idx] = f"Error: {str(e)}"
                        else:
                            results_dict[idx] = AnnotationResult(
                                annotation_graph=AnnotationGraph(),
                                overlay_image=None,
                                formal_report=f"Error: {str(e)}",
                                json_output={"error": str(e)},
                                processing_time=0.0
                            )
                
                # Sort by index to maintain order
                results = [results_dict[i] for i in sorted(results_dict.keys())]
        else:
            # Sequential processing
            for i, inp in enumerate(iterator):
                result = self._annotate_single_with_error_handling(inp, i, output_format)
                results.append(result)
        
        return results
    
    def _annotate_single_with_error_handling(
        self,
        input: Any,
        index: int,
        output_format: str
    ) -> Union[AnnotationResult, np.ndarray, Dict[str, Any], str]:
        """Annotate single input with error handling."""
        try:
            # Auto-load input
            image = self._auto_load_input(input)
            
            # Auto-detect type and tune
            image_type = None
            if self.config.auto_detect_type:
                image_type = self._detect_image_type(image)
            
            params = self._auto_tune_params(image, image_type)
            if image_type:
                params["detected_type"] = image_type
            
            # Annotate with retry
            result = self._annotate_with_retry(image, frame_id=index, params=params)
            
            return self._format_output(result, output_format)
        
        except Exception as e:
            logger.error(f"Error processing input {index}: {e}")
            # Return appropriate error result
            if output_format == "overlay":
                return np.zeros((100, 100, 3), dtype=np.uint8)
            elif output_format == "json":
                return {"error": str(e), "index": index}
            elif output_format == "report":
                return f"Error processing input {index}: {str(e)}"
            else:
                return AnnotationResult(
                    annotation_graph=AnnotationGraph(),
                    overlay_image=None,
                    formal_report=f"Error: {str(e)}",
                    json_output={"error": str(e), "index": index},
                    processing_time=0.0
                )
    
    def annotate_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image, Path]],
        frame_id: Optional[int] = None,
        output: Optional[str] = None
    ) -> List:
        """
        Annotate a batch of images (explicit batch method).
        
        Note: The main annotate() method auto-detects batch processing,
        so this method is mainly for explicit batch calls.
        
        Args:
            images: List of input images (any supported type)
            frame_id: Optional starting frame ID
            output: Output format (overrides config)
            
        Returns:
            List of results (format depends on output parameter)
        """
        return self._annotate_batch_auto(images, frame_id, output or self.config.output_format)
    
    def reset_temporal_state(self) -> None:
        """Reset temporal tracking state."""
        self._entity_trackers.clear()
        self._frame_history.clear()
        self._entity_id_map.clear()
        logger.info("Temporal state reset")
    
    # ==================== Query/Task-Based Interface ====================
    
    def query(
        self,
        input: Union[str, np.ndarray, Image.Image, Path],
        query: str,
        frame_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Answer a specific query about an image using natural language.
        
        This method performs annotation first, then uses the CR-CA agent to interpret
        the query and analyze the annotation graph to provide a structured answer.
        
        Examples:
            - "find the largest building in this image"
            - "measure the height of the tallest structure"
            - "identify all circles and calculate their total area"
            - "find the longest line and measure its length"
            - "count how many buildings are in this cityscape"
            - "find the largest building in this city and measure its dimensions"
        
        Usage Example:
            ```python
            from image_annotation import ImageAnnotationEngine
            
            engine = ImageAnnotationEngine()
            
            # Query: Find largest building and measure it
            result = engine.query(
                "cityscape.jpg",
                "find the largest building in this image and measure its dimensions"
            )
            
            print(result["answer"])
            # Output: "Found 1 matching entities. The largest entity is abc123 with size 12500.00 pixels².
            #          Measurements:
            #            Entity abc123: area=12500.00px², width=150.00px, height=83.33px"
            
            # Access entities
            for entity in result["entities"]:
                print(f"Entity {entity['id']}: {entity['label']}")
            
            # Access measurements
            for entity_id, measurements in result["measurements"].items():
                print(f"{entity_id}: {measurements}")
            
            # Save visualization
            if result["visualization"] is not None:
                cv2.imwrite("query_result.png", result["visualization"])
            ```
        
        Args:
            input: Image input (file path, URL, numpy array, PIL Image)
            query: Natural language query about the image
            frame_id: Optional frame ID for temporal tracking
            
        Returns:
            Dictionary with:
            - "answer": Natural language answer to the query
            - "entities": List of relevant entities found (with id, type, label, pixel_coords, metadata)
            - "measurements": Dict of measurements performed (area, length, width, height, etc.)
            - "confidence": Confidence score (0.0-1.0)
            - "reasoning": Step-by-step reasoning from CR-CA agent
            - "visualization": Optional overlay image (numpy array) highlighting relevant entities
            - "annotation_graph": Full AnnotationGraph object for advanced analysis
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Annotate the image first
        annotation_result = self.annotate(input, frame_id=frame_id, output="all")
        
        if not isinstance(annotation_result, AnnotationResult):
            # Fallback if annotate returned something else
            logger.warning("Annotation returned unexpected format, re-annotating...")
            image = self._auto_load_input(input)
            annotation_result = self._annotate_with_retry(image, frame_id, self._auto_tune_params(image))
        
        # Step 2: Use CR-CA agent to interpret query and analyze annotation graph
        query_result = self._process_query(query, annotation_result)
        
        # Step 3: Extract relevant entities based on query
        relevant_entities = self._extract_relevant_entities(query, annotation_result.annotation_graph)
        
        # Step 4: Perform measurements if requested
        measurements = self._perform_query_measurements(query, relevant_entities, annotation_result.annotation_graph)
        
        # Step 5: Generate visualization highlighting relevant entities
        visualization = self._generate_query_visualization(
            self._auto_load_input(input) if not isinstance(input, np.ndarray) else input,
            annotation_result.annotation_graph,
            relevant_entities
        )
        
        # Step 6: Compile final answer
        answer = self._generate_query_answer(query, query_result, relevant_entities, measurements)
        
        return {
            "answer": answer,
            "entities": [
                {
                    "id": e.id,
                    "type": e.primitive_type,
                    "label": self._get_entity_label(e.id, annotation_result.annotation_graph),
                    "pixel_coords": e.pixel_coords,
                    "metadata": e.metadata
                }
                for e in relevant_entities
            ],
            "measurements": measurements,
            "confidence": query_result.get("confidence", 0.5),
            "reasoning": query_result.get("reasoning", ""),
            "visualization": visualization,
            "annotation_graph": annotation_result.annotation_graph
        }
    
    def _process_query(self, query: str, annotation_result: AnnotationResult) -> Dict[str, Any]:
        """Use CR-CA agent to interpret query and analyze annotation graph."""
        # Prepare context for the agent
        graph_summary = self._summarize_annotation_graph(annotation_result.annotation_graph)
        
        query_prompt = f"""You are analyzing an annotated image to answer a specific query.

ANNOTATION SUMMARY:
{graph_summary}

USER QUERY: {query}

Your task is to:
1. Understand what the user is asking for
2. Identify which entities in the annotation graph are relevant
3. Determine what measurements or analysis are needed
4. Provide reasoning for your answer

Consider:
- Entity types (lines, circles, contours, intersections)
- Semantic labels (what each entity represents)
- Spatial relationships between entities
- Size/scale comparisons
- Counting operations
- Measurement requirements

Provide your analysis in a structured format:
- Relevant entity IDs
- Required measurements
- Reasoning steps
- Confidence level (0.0-1.0)
"""
        
        try:
            # Use CR-CA agent to process query
            response = self._crca_agent.run(task=query_prompt)
            
            # Parse response
            if isinstance(response, dict):
                return {
                    "reasoning": response.get("response", str(response)),
                    "confidence": 0.7,
                    "raw_response": response
                }
            else:
                return {
                    "reasoning": str(response),
                    "confidence": 0.7,
                    "raw_response": response
                }
        except Exception as e:
            logger.error(f"Error processing query with CR-CA agent: {e}")
            return {
                "reasoning": f"Query processing encountered an error: {str(e)}",
                "confidence": 0.3,
                "raw_response": None
            }
    
    def _summarize_annotation_graph(self, graph: AnnotationGraph) -> str:
        """Create a summary of the annotation graph for query processing."""
        summary_lines = []
        
        summary_lines.append(f"Total entities: {len(graph.entities)}")
        summary_lines.append(f"Total labels: {len(graph.labels)}")
        summary_lines.append(f"Total relations: {len(graph.relations)}")
        summary_lines.append(f"Contradictions: {len(graph.contradictions)}")
        
        # Entity type breakdown
        type_counts = {}
        for entity in graph.entities:
            type_counts[entity.primitive_type] = type_counts.get(entity.primitive_type, 0) + 1
        summary_lines.append(f"Entity types: {dict(type_counts)}")
        
        # Sample labels
        if graph.labels:
            summary_lines.append("\nSample labels:")
            for label in graph.labels[:10]:  # First 10 labels
                entity = graph.get_entity_by_id(label.entity_id)
                if entity:
                    summary_lines.append(f"  - {label.label} (entity {label.entity_id}, type: {entity.primitive_type}, uncertainty: {label.uncertainty:.2f})")
        
        # Relations
        if graph.relations:
            summary_lines.append("\nRelations:")
            for relation in graph.relations[:10]:  # First 10 relations
                summary_lines.append(f"  - {relation.source_id} --[{relation.relation_type}]--> {relation.target_id}")
        
        return "\n".join(summary_lines)
    
    def _extract_relevant_entities(
        self,
        query: str,
        graph: AnnotationGraph
    ) -> List[PrimitiveEntity]:
        """Extract entities relevant to the query."""
        query_lower = query.lower()
        relevant_entities = []
        
        # Keywords that suggest entity types
        if "building" in query_lower or "structure" in query_lower:
            # Look for entities labeled as buildings
            for label in graph.labels:
                if "building" in label.label.lower() or "structure" in label.label.lower():
                    entity = graph.get_entity_by_id(label.entity_id)
                    if entity:
                        relevant_entities.append(entity)
        
        if "circle" in query_lower or "round" in query_lower:
            for entity in graph.entities:
                if entity.primitive_type == "circle":
                    relevant_entities.append(entity)
        
        if "line" in query_lower or "edge" in query_lower:
            for entity in graph.entities:
                if entity.primitive_type == "line":
                    relevant_entities.append(entity)
        
        # Size-related queries
        if "largest" in query_lower or "biggest" in query_lower or "tallest" in query_lower:
            # Find largest entity by area/radius
            if relevant_entities:
                # Filter to largest
                largest = max(relevant_entities, key=lambda e: self._get_entity_size(e))
                relevant_entities = [largest]
            else:
                # Find largest overall
                if graph.entities:
                    largest = max(graph.entities, key=lambda e: self._get_entity_size(e))
                    relevant_entities = [largest]
        
        if "smallest" in query_lower or "tiny" in query_lower:
            if relevant_entities:
                smallest = min(relevant_entities, key=lambda e: self._get_entity_size(e))
                relevant_entities = [smallest]
            else:
                if graph.entities:
                    smallest = min(graph.entities, key=lambda e: self._get_entity_size(e))
                    relevant_entities = [smallest]
        
        # Count queries - return all matching entities
        if "count" in query_lower or "how many" in query_lower:
            # Return all relevant entities for counting
            pass  # Already collected above
        
        # If no specific matches, return all entities
        if not relevant_entities:
            relevant_entities = graph.entities[:20]  # Limit to first 20 for performance
        
        return relevant_entities
    
    def _get_entity_size(self, entity: PrimitiveEntity) -> float:
        """Calculate size metric for entity (area, length, etc.)."""
        if entity.primitive_type == "circle":
            radius = entity.metadata.get("radius", 0)
            return np.pi * radius * radius  # Area
        elif entity.primitive_type == "line":
            if len(entity.pixel_coords) >= 2:
                p1, p2 = entity.pixel_coords[0], entity.pixel_coords[1]
                return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  # Length
        elif entity.primitive_type == "contour":
            if len(entity.pixel_coords) >= 3:
                # Calculate polygon area using shoelace formula
                points = np.array(entity.pixel_coords)
                x = points[:, 0]
                y = points[:, 1]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        return 0.0
    
    def _perform_query_measurements(
        self,
        query: str,
        entities: List[PrimitiveEntity],
        graph: AnnotationGraph
    ) -> Dict[str, Any]:
        """Perform measurements requested by the query."""
        query_lower = query.lower()
        measurements = {}
        
        # Check if measurement is requested
        if "measure" in query_lower or "size" in query_lower or "height" in query_lower or "width" in query_lower or "area" in query_lower or "length" in query_lower:
            for entity in entities:
                entity_id = entity.id
                
                if entity.primitive_type == "circle":
                    radius = entity.metadata.get("radius", 0)
                    measurements[entity_id] = {
                        "type": "circle",
                        "radius": float(radius),
                        "diameter": float(radius * 2),
                        "area": float(np.pi * radius * radius),
                        "circumference": float(2 * np.pi * radius)
                    }
                
                elif entity.primitive_type == "line":
                    if len(entity.pixel_coords) >= 2:
                        p1, p2 = entity.pixel_coords[0], entity.pixel_coords[1]
                        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        measurements[entity_id] = {
                            "type": "line",
                            "length": float(length),
                            "start_point": p1,
                            "end_point": p2
                        }
                
                elif entity.primitive_type == "contour":
                    if len(entity.pixel_coords) >= 3:
                        points = np.array(entity.pixel_coords)
                        x = points[:, 0]
                        y = points[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        
                        # Bounding box
                        x_min, x_max = int(np.min(x)), int(np.max(x))
                        y_min, y_max = int(np.min(y)), int(np.max(y))
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        measurements[entity_id] = {
                            "type": "contour",
                            "area": float(area),
                            "width": float(width),
                            "height": float(height),
                            "bounding_box": {
                                "x_min": int(x_min),
                                "y_min": int(y_min),
                                "x_max": int(x_max),
                                "y_max": int(y_max)
                            }
                        }
        
        return measurements
    
    def _get_entity_label(self, entity_id: str, graph: AnnotationGraph) -> Optional[str]:
        """Get label for entity."""
        labels = graph.get_labels_for_entity(entity_id)
        if labels:
            return labels[0].label
        return None
    
    def _generate_query_visualization(
        self,
        image: np.ndarray,
        graph: AnnotationGraph,
        relevant_entities: List[PrimitiveEntity]
    ) -> Optional[np.ndarray]:
        """Generate visualization highlighting relevant entities."""
        try:
            # Create overlay with highlighted relevant entities
            overlay = image.copy()
            
            # Highlight relevant entities in different color
            for entity in relevant_entities:
                if entity.primitive_type == "line" and len(entity.pixel_coords) >= 2:
                    cv2.line(overlay, entity.pixel_coords[0], entity.pixel_coords[1], (0, 255, 0), 3)
                elif entity.primitive_type == "circle" and len(entity.pixel_coords) >= 1:
                    center = entity.pixel_coords[0]
                    radius = int(entity.metadata.get("radius", 10))
                    cv2.circle(overlay, center, radius, (0, 255, 0), 3)
                elif entity.primitive_type == "contour" and len(entity.pixel_coords) >= 3:
                    points = np.array(entity.pixel_coords, dtype=np.int32)
                    cv2.polylines(overlay, [points], True, (0, 255, 0), 3)
            
            return overlay
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None
    
    def _generate_query_answer(
        self,
        query: str,
        query_result: Dict[str, Any],
        entities: List[PrimitiveEntity],
        measurements: Dict[str, Any]
    ) -> str:
        """Generate natural language answer to the query."""
        answer_parts = []
        
        # Count queries
        if "count" in query.lower() or "how many" in query.lower():
            answer_parts.append(f"Found {len(entities)} matching entities.")
        
        # Size queries
        if "largest" in query.lower() or "biggest" in query.lower():
            if entities:
                entity = entities[0]
                size = self._get_entity_size(entity)
                answer_parts.append(f"The largest entity is {entity.id} with size {size:.2f} pixels².")
        
        # Measurement queries
        if measurements:
            answer_parts.append("\nMeasurements:")
            for entity_id, meas in measurements.items():
                if meas["type"] == "circle":
                    answer_parts.append(f"  Entity {entity_id}: radius={meas['radius']:.2f}px, area={meas['area']:.2f}px²")
                elif meas["type"] == "line":
                    answer_parts.append(f"  Entity {entity_id}: length={meas['length']:.2f}px")
                elif meas["type"] == "contour":
                    answer_parts.append(f"  Entity {entity_id}: area={meas['area']:.2f}px², width={meas['width']:.2f}px, height={meas['height']:.2f}px")
        
        # Add reasoning if available
        if query_result.get("reasoning"):
            answer_parts.append(f"\nReasoning: {query_result['reasoning']}")
        
        if not answer_parts:
            answer_parts.append("Query processed, but no specific answer could be generated.")
        
        return "\n".join(answer_parts)
