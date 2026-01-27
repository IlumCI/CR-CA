"""
Unit tests for image annotation engine - testing every feature individually.
"""

import os
import sys
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import tempfile
import json
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Image annotation unit tests can be slow and may require LLM access depending on configuration.
if os.environ.get("CRCA_RUN_IMAGE_ANNOTATION_TESTS") != "1":
    pytest.skip("Set CRCA_RUN_IMAGE_ANNOTATION_TESTS=1 to run image annotation tests", allow_module_level=True)

try:
    from image_annotation.annotation_engine import ImageAnnotationEngine, AnnotationConfig
    from image_annotation import AnnotationResult, AnnotationGraph, PrimitiveEntity, SemanticLabel
    IMAGE_ANNOTATION_AVAILABLE = True
except ImportError:
    IMAGE_ANNOTATION_AVAILABLE = False
    pytest.skip("Image annotation not available", allow_module_level=True)


class TestImageAnnotationConfig:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AnnotationConfig()
        assert config.gpt_model == "gpt-4o-mini"
        assert config.cache_enabled is True
        assert config.auto_retry is True
        assert config.output_format == "overlay"
    
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        import os
        os.environ["ANNOTATION_GPT_MODEL"] = "gpt-4"
        os.environ["ANNOTATION_CACHE_ENABLED"] = "false"
        
        config = AnnotationConfig.from_env()
        assert config.gpt_model == "gpt-4"
        assert config.cache_enabled is False
        
        # Cleanup
        del os.environ["ANNOTATION_GPT_MODEL"]
        del os.environ["ANNOTATION_CACHE_ENABLED"]
    
    def test_config_override(self):
        """Test configuration parameter override."""
        config = AnnotationConfig(gpt_model="gpt-4o", cache_enabled=False)
        assert config.gpt_model == "gpt-4o"
        assert config.cache_enabled is False


class TestInputHandling:
    """Test smart input handling features."""
    
    def test_detect_input_type_file_path(self):
        """Test file path detection."""
        engine = ImageAnnotationEngine()
        assert engine._detect_input_type("image.png") == "file_path"
        assert engine._detect_input_type(Path("image.png")) == "file_path"
    
    def test_detect_input_type_url(self):
        """Test URL detection."""
        engine = ImageAnnotationEngine()
        assert engine._detect_input_type("https://example.com/image.png") == "url"
        assert engine._detect_input_type("http://example.com/image.png") == "url"
    
    def test_detect_input_type_numpy(self):
        """Test numpy array detection."""
        engine = ImageAnnotationEngine()
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        assert engine._detect_input_type(arr) == "numpy_array"
    
    def test_detect_input_type_pil(self):
        """Test PIL Image detection."""
        engine = ImageAnnotationEngine()
        img = Image.new("RGB", (100, 100))
        assert engine._detect_input_type(img) == "pil_image"
    
    def test_detect_input_type_batch(self):
        """Test batch detection."""
        engine = ImageAnnotationEngine()
        assert engine._detect_input_type(["img1.png", "img2.png"]) == "batch"
    
    def test_auto_load_numpy(self):
        """Test loading numpy array."""
        engine = ImageAnnotationEngine()
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        loaded = engine._auto_load_input(arr)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == (100, 100, 3)
    
    def test_auto_load_pil(self):
        """Test loading PIL Image."""
        engine = ImageAnnotationEngine()
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        loaded = engine._auto_load_input(img)
        assert isinstance(loaded, np.ndarray)
        assert len(loaded.shape) == 3  # BGR format
    
    def test_auto_load_file(self):
        """Test loading from file path."""
        engine = ImageAnnotationEngine()
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (100, 100), color=(0, 255, 0))
            img.save(f.name, "PNG")
            temp_path = f.name
        
        try:
            loaded = engine._auto_load_input(temp_path)
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape[0] > 0 and loaded.shape[1] > 0
        finally:
            os.unlink(temp_path)


class TestImageTypeDetection:
    """Test image type detection."""
    
    def test_detect_circuit(self):
        """Test circuit diagram detection."""
        engine = ImageAnnotationEngine()
        # Create synthetic circuit-like image (high line density, circles)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        # Add lines
        for i in range(0, 500, 20):
            cv2.line(img, (i, 0), (i, 500), (255, 255, 255), 2)
        # Add circles
        for i in range(50, 450, 50):
            cv2.circle(img, (i, i), 20, (255, 255, 255), 2)
        
        img_type = engine._detect_image_type(img)
        # Should detect as circuit or general
        assert img_type in ["circuit", "general", "technical"]
    
    def test_detect_architectural(self):
        """Test architectural drawing detection."""
        engine = ImageAnnotationEngine()
        # Create synthetic architectural drawing (many parallel lines)
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        # Add many parallel lines
        for i in range(0, 800, 10):
            cv2.line(img, (0, i), (800, i), (255, 255, 255), 1)
            cv2.line(img, (i, 0), (i, 800), (255, 255, 255), 1)
        
        img_type = engine._detect_image_type(img)
        # Should detect as architectural or general
        assert img_type in ["architectural", "general", "technical"]
    
    def test_get_type_specific_params(self):
        """Test getting type-specific parameters."""
        engine = ImageAnnotationEngine()
        params = engine._get_type_specific_params("circuit")
        assert "hough_line_threshold" in params
        assert "hough_circle_threshold" in params
        assert "preprocessing_strength" in params


class TestParameterTuning:
    """Test automatic parameter tuning."""
    
    def test_auto_tune_params(self):
        """Test automatic parameter tuning."""
        engine = ImageAnnotationEngine()
        img = np.zeros((500, 500), dtype=np.uint8)
        # Add some edges
        cv2.rectangle(img, (100, 100), (400, 400), 255, 2)
        
        params = engine._auto_tune_params(img)
        assert isinstance(params, dict)
        assert "hough_line_threshold" in params
        assert "canny_low" in params
        assert "canny_high" in params
    
    def test_heuristic_tune(self):
        """Test heuristic-based tuning."""
        engine = ImageAnnotationEngine()
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, 255, 2)
        
        params = engine._heuristic_tune(img, "circuit")
        assert isinstance(params, dict)
        assert params["hough_line_threshold"] > 0
    
    def test_adaptive_refine(self):
        """Test adaptive parameter refinement."""
        engine = ImageAnnotationEngine()
        img = np.zeros((500, 500), dtype=np.uint8)
        
        # Create empty result (no primitives)
        from schemas.annotation import AnnotationGraph, AnnotationResult
        empty_result = AnnotationResult(
            annotation_graph=AnnotationGraph(),
            overlay_image=None,
            formal_report="",
            json_output={},
            processing_time=0.0
        )
        
        base_params = {"hough_line_threshold": 100, "hough_circle_threshold": 30}
        refined = engine._adaptive_refine(img, empty_result, base_params)
        # Should relax thresholds
        assert refined["hough_line_threshold"] <= base_params["hough_line_threshold"]


class TestRetryLogic:
    """Test retry logic."""
    
    def test_should_retry_no_primitives(self):
        """Test retry decision for no primitives."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import AnnotationGraph, AnnotationResult
        result = AnnotationResult(
            annotation_graph=AnnotationGraph(),  # Empty
            overlay_image=None,
            formal_report="",
            json_output={},
            processing_time=0.0
        )
        
        assert engine._should_retry(result, attempt=0) is True
    
    def test_should_retry_low_confidence(self):
        """Test retry decision for low confidence."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import AnnotationGraph, AnnotationResult, SemanticLabel, PrimitiveEntity
        
        entity = PrimitiveEntity(id="test", pixel_coords=[(0, 0)], primitive_type="line")
        label = SemanticLabel(entity_id="test", label="test", uncertainty=0.9)  # High uncertainty
        
        graph = AnnotationGraph(entities=[entity], labels=[label])
        result = AnnotationResult(
            annotation_graph=graph,
            overlay_image=None,
            formal_report="",
            json_output={},
            processing_time=0.0
        )
        
        assert engine._should_retry(result, attempt=0) is True
    
    def test_should_retry_max_attempts(self):
        """Test retry decision when max attempts reached."""
        engine = ImageAnnotationEngine()
        engine.config.max_retries = 3
        
        from schemas.annotation import AnnotationGraph, AnnotationResult
        result = AnnotationResult(
            annotation_graph=AnnotationGraph(),
            overlay_image=None,
            formal_report="",
            json_output={},
            processing_time=0.0
        )
        
        assert engine._should_retry(result, attempt=3) is False
    
    def test_get_retry_params(self):
        """Test getting retry parameters."""
        engine = ImageAnnotationEngine()
        base_params = {"hough_line_threshold": 100, "canny_low": 50}
        
        retry_params = engine._get_retry_params(attempt=1, previous_result=None, base_params=base_params)
        # Should relax thresholds
        assert retry_params["hough_line_threshold"] < base_params["hough_line_threshold"]


class TestCaching:
    """Test smart caching."""
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        engine = ImageAnnotationEngine()
        # Create actually different images
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1[50, 50] = [255, 255, 255]  # Add a white pixel
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[50, 50] = [128, 128, 128]  # Add a gray pixel (different)
        params = {"test": "value"}
        
        key1 = engine._get_cache_key(img1, params)
        key2 = engine._get_cache_key(img2, params)
        key3 = engine._get_cache_key(img1, params)
        
        # Same image + params should give same key
        assert key1 == key3
        # Different images should give different keys (even if same shape)
        assert key1 != key2
    
    def test_cache_primitives(self):
        """Test caching primitives."""
        engine = ImageAnnotationEngine()
        engine.config.cache_enabled = True
        
        from schemas.annotation import PrimitiveEntity
        primitives = [
            PrimitiveEntity(id="1", pixel_coords=[(0, 0), (10, 10)], primitive_type="line")
        ]
        
        cache_key = "test_key"
        engine._cache_primitives(cache_key, primitives)
        
        # Check cache file exists
        cache_file = engine._cache_dir / f"{cache_key}_primitives.json"
        assert cache_file.exists()
        
        # Cleanup
        if cache_file.exists():
            cache_file.unlink()
    
    def test_get_cached_primitives(self):
        """Test retrieving cached primitives."""
        engine = ImageAnnotationEngine()
        engine.config.cache_enabled = True
        
        from schemas.annotation import PrimitiveEntity
        primitives = [
            PrimitiveEntity(id="1", pixel_coords=[(0, 0), (10, 10)], primitive_type="line")
        ]
        
        cache_key = "test_key_2"
        engine._cache_primitives(cache_key, primitives)
        
        cached = engine._get_cached_primitives(cache_key)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].id == "1"
        
        # Cleanup
        cache_file = engine._cache_dir / f"{cache_key}_primitives.json"
        if cache_file.exists():
            cache_file.unlink()


class TestPrimitiveExtraction:
    """Test primitive extraction methods."""
    
    def test_extract_lines(self):
        """Test line extraction."""
        engine = ImageAnnotationEngine()
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.line(img, (0, 0), (500, 500), 255, 2)
        cv2.line(img, (0, 500), (500, 0), 255, 2)
        
        params = {"hough_line_threshold": 50, "hough_line_min_length": 10}
        lines = engine._extract_lines(img, params)
        assert len(lines) > 0
    
    def test_extract_circles(self):
        """Test circle extraction."""
        engine = ImageAnnotationEngine()
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(img, (250, 250), 50, 255, 2)
        
        params = {"hough_circle_threshold": 30, "hough_circle_min_radius": 10}
        circles = engine._extract_circles(img, params)
        assert len(circles) > 0
    
    def test_extract_contours(self):
        """Test contour extraction."""
        engine = ImageAnnotationEngine()
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, 2)
        
        params = {"canny_low": 50, "canny_high": 150}
        contours = engine._extract_contours(img, params)
        assert len(contours) > 0
    
    def test_compute_intersections(self):
        """Test intersection computation."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import PrimitiveEntity
        
        # Create two lines that intersect
        line1 = PrimitiveEntity(
            id="line1",
            pixel_coords=[(0, 0), (100, 100)],
            primitive_type="line"
        )
        line2 = PrimitiveEntity(
            id="line2",
            pixel_coords=[(0, 100), (100, 0)],
            primitive_type="line"
        )
        
        intersections = engine._compute_intersections([line1, line2])
        assert len(intersections) > 0


class TestPreprocessing:
    """Test image preprocessing."""
    
    def test_preprocess_image(self):
        """Test main preprocessing pipeline."""
        engine = ImageAnnotationEngine()
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        
        processed = engine._preprocess_image(img)
        assert processed.shape[0] > 0
        assert processed.shape[1] > 0
        assert len(processed.shape) == 2  # Grayscale
    
    def test_adaptive_histogram_equalization(self):
        """Test adaptive histogram equalization."""
        engine = ImageAnnotationEngine()
        img = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        
        equalized = engine._adaptive_histogram_equalization(img)
        assert equalized.shape == img.shape
    
    def test_edge_amplification(self):
        """Test edge amplification."""
        engine = ImageAnnotationEngine()
        img = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        
        amplified = engine._edge_amplification(img, strength=0.7)
        assert amplified.shape == img.shape


class TestQueryInterface:
    """Test query/task-based interface."""
    
    def test_extract_relevant_entities(self):
        """Test extracting relevant entities from query."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import AnnotationGraph, PrimitiveEntity, SemanticLabel
        
        entity1 = PrimitiveEntity(id="1", pixel_coords=[(0, 0)], primitive_type="circle")
        entity2 = PrimitiveEntity(id="2", pixel_coords=[(10, 10), (20, 20)], primitive_type="line")
        label = SemanticLabel(entity_id="1", label="building", uncertainty=0.3)
        
        graph = AnnotationGraph(entities=[entity1, entity2], labels=[label])
        
        relevant = engine._extract_relevant_entities("find the largest building", graph)
        assert len(relevant) > 0
    
    def test_get_entity_size(self):
        """Test entity size calculation."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import PrimitiveEntity
        
        # Circle
        circle = PrimitiveEntity(
            id="circle1",
            pixel_coords=[(50, 50)],
            primitive_type="circle",
            metadata={"radius": 10}
        )
        size = engine._get_entity_size(circle)
        assert size > 0
        
        # Line
        line = PrimitiveEntity(
            id="line1",
            pixel_coords=[(0, 0), (10, 10)],
            primitive_type="line"
        )
        size = engine._get_entity_size(line)
        assert size > 0
    
    def test_perform_query_measurements(self):
        """Test performing measurements for queries."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import PrimitiveEntity, AnnotationGraph
        
        circle = PrimitiveEntity(
            id="circle1",
            pixel_coords=[(50, 50)],
            primitive_type="circle",
            metadata={"radius": 10}
        )
        graph = AnnotationGraph(entities=[circle])
        
        measurements = engine._perform_query_measurements("measure the circle", [circle], graph)
        assert "circle1" in measurements
        assert "radius" in measurements["circle1"]
        assert "area" in measurements["circle1"]


class TestOutputFormatting:
    """Test output formatting."""
    
    def test_format_output_overlay(self):
        """Test formatting output as overlay."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import AnnotationGraph, AnnotationResult
        import cv2
        
        overlay_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', overlay_img)
        overlay_bytes = buffer.tobytes()
        
        result = AnnotationResult(
            annotation_graph=AnnotationGraph(),
            overlay_image=overlay_bytes,
            formal_report="Test report",
            json_output={"test": "data"},
            processing_time=1.0
        )
        
        formatted = engine._format_output(result, "overlay")
        assert isinstance(formatted, np.ndarray)
    
    def test_format_output_json(self):
        """Test formatting output as JSON."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import AnnotationGraph, AnnotationResult
        
        result = AnnotationResult(
            annotation_graph=AnnotationGraph(),
            overlay_image=None,
            formal_report="Test report",
            json_output={"test": "data"},
            processing_time=1.0
        )
        
        formatted = engine._format_output(result, "json")
        assert isinstance(formatted, dict)
        assert "test" in formatted
    
    def test_format_output_report(self):
        """Test formatting output as report."""
        engine = ImageAnnotationEngine()
        from schemas.annotation import AnnotationGraph, AnnotationResult
        
        result = AnnotationResult(
            annotation_graph=AnnotationGraph(),
            overlay_image=None,
            formal_report="Test report content",
            json_output={},
            processing_time=1.0
        )
        
        formatted = engine._format_output(result, "report")
        assert isinstance(formatted, str)
        assert "Test report" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
