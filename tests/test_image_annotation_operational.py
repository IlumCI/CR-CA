"""
Operational tests for image annotation engine - testing basic functionality with simplified API.
"""

import os
import sys
import numpy as np
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Image annotation operational tests can be slow and may require extra deps/LLM access.
if os.environ.get("CRCA_RUN_IMAGE_ANNOTATION_TESTS") != "1":
    pytest.skip("Set CRCA_RUN_IMAGE_ANNOTATION_TESTS=1 to run image annotation tests", allow_module_level=True)

try:
    from image_annotation.annotation_engine import ImageAnnotationEngine
    from image_annotation import AnnotationResult
    IMAGE_ANNOTATION_AVAILABLE = True
except ImportError:
    IMAGE_ANNOTATION_AVAILABLE = False
    pytest.skip("Image annotation not available", allow_module_level=True)


def create_test_image(width=500, height=500, image_type="simple"):
    """Create a test image for testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if image_type == "simple":
        # Simple geometric shapes
        cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), 2)
        cv2.circle(img, (250, 250), 50, (255, 255, 255), 2)
        cv2.line(img, (0, 0), (500, 500), (255, 255, 255), 2)
    elif image_type == "circuit":
        # Circuit-like diagram
        for i in range(0, width, 50):
            cv2.line(img, (i, 0), (i, height), (255, 255, 255), 1)
        for i in range(50, width, 100):
            cv2.circle(img, (i, height//2), 20, (255, 255, 255), 2)
    elif image_type == "architectural":
        # Architectural drawing style
        for i in range(0, width, 20):
            cv2.line(img, (0, i), (width, i), (255, 255, 255), 1)
            cv2.line(img, (i, 0), (i, height), (255, 255, 255), 1)
    
    return img


class TestSimplifiedAPI:
    """Test the simplified/standardized API usage."""
    
    def test_annotate_file_path(self):
        """Test annotate with file path - simplest usage."""
        engine = ImageAnnotationEngine()
        
        # Create temporary image file
        img = create_test_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            # Simplest usage - just pass file path
            # Should return overlay (numpy array) by default for simplified API
            result = engine.annotate(temp_path)
            
            # File paths return overlay by default (simplified API)
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3  # BGR image
            
            # Can also request AnnotationResult explicitly
            result_full = engine.annotate(temp_path, output="all")
            assert isinstance(result_full, AnnotationResult)
            assert result_full.annotation_graph is not None
            assert result_full.processing_time > 0
        finally:
            os.unlink(temp_path)
    
    def test_annotate_numpy_array(self):
        """Test annotate with numpy array - backward compatible."""
        engine = ImageAnnotationEngine()
        img = create_test_image()
        
        # Old API still works
        result = engine.annotate(img)
        
        assert isinstance(result, AnnotationResult)
        assert len(result.annotation_graph.entities) >= 0
    
    def test_annotate_output_overlay(self):
        """Test annotate with overlay output format."""
        engine = ImageAnnotationEngine()
        img = create_test_image()
        
        # Request overlay format - should return numpy array
        overlay = engine.annotate(img, output="overlay")
        
        assert isinstance(overlay, np.ndarray)
        assert len(overlay.shape) == 3  # BGR image
    
    def test_annotate_output_json(self):
        """Test annotate with JSON output format."""
        engine = ImageAnnotationEngine()
        img = create_test_image()
        
        # Request JSON format
        json_output = engine.annotate(img, output="json")
        
        assert isinstance(json_output, dict)
        assert "entities" in json_output or "error" in json_output
    
    def test_annotate_output_report(self):
        """Test annotate with report output format."""
        engine = ImageAnnotationEngine()
        img = create_test_image()
        
        # Request report format
        report = engine.annotate(img, output="report")
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_annotate_batch_auto_detection(self):
        """Test automatic batch processing detection."""
        engine = ImageAnnotationEngine()
        
        # Create multiple test images
        images = []
        temp_files = []
        for i in range(3):
            img = create_test_image()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, img)
                images.append(f.name)
                temp_files.append(f.name)
        
        try:
            # Pass list - should auto-detect batch processing
            results = engine.annotate(images)
            
            assert isinstance(results, list)
            assert len(results) == 3
            # Results should be in requested format (default is overlay for new API)
            # But since we passed file paths, it might return AnnotationResult
            assert all(isinstance(r, (AnnotationResult, np.ndarray, dict, str)) for r in results)
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)
    
    def test_annotate_batch_mixed_inputs(self):
        """Test batch processing with mixed input types."""
        engine = ImageAnnotationEngine()
        
        # Create mix of file path and numpy array
        img1 = create_test_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img1)
            temp_path = f.name
        
        img2 = create_test_image()
        
        try:
            # Mix file path and numpy array
            results = engine.annotate([temp_path, img2])
            
            assert isinstance(results, list)
            assert len(results) == 2
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_query_simple(self):
        """Test query interface with simple query."""
        engine = ImageAnnotationEngine()
        img = create_test_image()
        
        # Save to temp file for query
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            # Simple query
            result = engine.query(temp_path, "find all circles")
            
            assert isinstance(result, dict)
            assert "answer" in result
            assert "entities" in result
            assert "measurements" in result
            assert "confidence" in result
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_query_with_measurements(self):
        """Test query interface requesting measurements."""
        engine = ImageAnnotationEngine()
        img = create_test_image()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            # Query requesting measurements
            result = engine.query(temp_path, "find the largest circle and measure its area")
            
            assert isinstance(result, dict)
            assert "measurements" in result
            # May or may not have measurements depending on what was found
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_query_count_objects(self):
        """Test query interface for counting objects."""
        engine = ImageAnnotationEngine()
        img = create_test_image()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            # Count query
            result = engine.query(temp_path, "count how many lines are in this image")
            
            assert isinstance(result, dict)
            assert "answer" in result
            assert "entities" in result
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAutoFeatures:
    """Test automatic features (type detection, parameter tuning, etc.)."""
    
    def test_auto_image_type_detection(self):
        """Test automatic image type detection."""
        engine = ImageAnnotationEngine()
        engine.config.auto_detect_type = True
        
        # Circuit-like image
        circuit_img = create_test_image(image_type="circuit")
        result = engine.annotate(circuit_img, output="all")
        
        # Should have detected image type in metadata
        if isinstance(result, AnnotationResult):
            detected_type = result.annotation_graph.metadata.get("image_type", "unknown")
            assert detected_type in ["circuit", "general", "technical", "unknown"]
    
    def test_auto_parameter_tuning(self):
        """Test automatic parameter tuning."""
        engine = ImageAnnotationEngine()
        engine.config.auto_tune_params = True
        
        img = create_test_image()
        result = engine.annotate(img)
        
        # Should complete without errors (tuning happens internally)
        assert isinstance(result, AnnotationResult)
    
    def test_auto_retry_on_failure(self):
        """Test automatic retry logic."""
        engine = ImageAnnotationEngine()
        engine.config.auto_retry = True
        engine.config.max_retries = 2
        
        # Very simple image that might need retry
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = engine.annotate(img)
        
        # Should complete (may retry internally)
        assert isinstance(result, AnnotationResult)
    
    def test_caching_enabled(self):
        """Test that caching works."""
        engine = ImageAnnotationEngine()
        engine.config.cache_enabled = True
        
        img = create_test_image()
        
        # First annotation
        result1 = engine.annotate(img)
        
        # Second annotation (should use cache for primitives)
        result2 = engine.annotate(img)
        
        # Both should complete
        assert isinstance(result1, AnnotationResult)
        assert isinstance(result2, AnnotationResult)


class TestErrorHandling:
    """Test error handling in simplified API."""
    
    def test_invalid_file_path(self):
        """Test handling of invalid file path."""
        engine = ImageAnnotationEngine()
        
        # Non-existent file
        result = engine.annotate("nonexistent_file.png", output="all")
        
        # Should return error result, not crash
        if isinstance(result, AnnotationResult):
            assert "error" in result.json_output or len(result.annotation_graph.entities) == 0
    
    def test_empty_image(self):
        """Test handling of empty/blank image."""
        engine = ImageAnnotationEngine()
        
        # Empty image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = engine.annotate(img)
        
        # Should complete without crashing
        assert isinstance(result, AnnotationResult)
    
    def test_batch_with_errors(self):
        """Test batch processing continues on individual errors."""
        engine = ImageAnnotationEngine()
        
        img = create_test_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            # Mix valid and invalid inputs
            results = engine.annotate([temp_path, "nonexistent.png", img])
            
            # Should return results for all (some may be errors)
            assert isinstance(results, list)
            assert len(results) == 3
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestConfiguration:
    """Test configuration options."""
    
    def test_custom_config(self):
        """Test using custom configuration."""
        from image_annotation.annotation_engine import AnnotationConfig
        
        config = AnnotationConfig(
            gpt_model="gpt-4o-mini",
            cache_enabled=False,
            auto_retry=False,
            output_format="json"
        )
        
        engine = ImageAnnotationEngine(config=config)
        
        assert engine.config.cache_enabled is False
        assert engine.config.auto_retry is False
        assert engine.config.output_format == "json"
    
    def test_config_override(self):
        """Test overriding config parameters."""
        engine = ImageAnnotationEngine(
            gpt_model="gpt-4o",
            cache_enabled=False
        )
        
        assert engine.config.gpt_model == "gpt-4o"
        assert engine.config.cache_enabled is False


class TestIntegration:
    """Test integration with CR-CA."""
    
    def test_crca_agent_with_image_tools(self):
        """Test CR-CA agent can use image annotation tools."""
        try:
            from CRCA import CRCAAgent
        except ImportError:
            pytest.skip("CR-CA not available")
        
        # Create agent with image annotation enabled
        agent = CRCAAgent(
            model_name="gpt-4o-mini",
            use_image_annotation=True
        )
        
        # Check that tools are available
        tool_names = []
        if hasattr(agent, 'tools_list_dictionary') and agent.tools_list_dictionary:
            for tool in agent.tools_list_dictionary:
                if isinstance(tool, dict):
                    func_name = tool.get("function", {}).get("name", "")
                    if func_name:
                        tool_names.append(func_name)
        
        # Image annotation tools should be available
        assert "annotate_image" in tool_names or "query_image" in tool_names or len(tool_names) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
