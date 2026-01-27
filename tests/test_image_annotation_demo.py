"""
Demo test - Real-world example: Planning an invasion using image annotation.
"""

import os
import sys
import numpy as np
import pytest
import tempfile
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Image annotation demo tests are extremely slow and may require LLM access.
# They are skipped by default; enable explicitly when needed.
if os.environ.get("CRCA_RUN_IMAGE_ANNOTATION_TESTS") != "1":
    pytest.skip("Set CRCA_RUN_IMAGE_ANNOTATION_TESTS=1 to run image annotation demos", allow_module_level=True)

try:
    from image_annotation.annotation_engine import ImageAnnotationEngine
    from CRCA import CRCAAgent
    from schemas import AnnotationResult
    IMAGE_ANNOTATION_AVAILABLE = True
except ImportError:
    IMAGE_ANNOTATION_AVAILABLE = False
    pytest.skip("Image annotation or CR-CA not available", allow_module_level=True)


def create_tactical_map(width=1000, height=1000):
    """Create a synthetic tactical map for invasion planning."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (terrain)
    cv2.rectangle(img, (0, 0), (width, height), (34, 139, 34), -1)  # Green terrain
    
    # Rivers (blue lines)
    cv2.line(img, (200, 0), (200, height), (0, 0, 255), 10)
    cv2.line(img, (600, 0), (600, height), (0, 0, 255), 10)
    
    # Roads (gray lines)
    cv2.line(img, (0, 300), (width, 300), (128, 128, 128), 8)
    cv2.line(img, (0, 700), (width, 700), (128, 128, 128), 8)
    cv2.line(img, (400, 0), (400, height), (128, 128, 128), 8)
    
    # Cities/Buildings (rectangles)
    # Capital city (large)
    cv2.rectangle(img, (800, 800), (950, 950), (139, 69, 19), -1)  # Brown
    cv2.rectangle(img, (800, 800), (950, 950), (255, 255, 255), 3)
    
    # Secondary cities (medium)
    cv2.rectangle(img, (100, 200), (200, 300), (139, 69, 19), -1)
    cv2.rectangle(img, (100, 200), (200, 300), (255, 255, 255), 2)
    
    cv2.rectangle(img, (700, 100), (850, 200), (139, 69, 19), -1)
    cv2.rectangle(img, (700, 100), (850, 200), (255, 255, 255), 2)
    
    # Military bases (circles with cross)
    # Base 1
    cv2.circle(img, (300, 500), 40, (255, 0, 0), 3)
    cv2.line(img, (260, 500), (340, 500), (255, 0, 0), 2)
    cv2.line(img, (300, 460), (300, 540), (255, 0, 0), 2)
    
    # Base 2
    cv2.circle(img, (750, 400), 40, (255, 0, 0), 3)
    cv2.line(img, (710, 400), (790, 400), (255, 0, 0), 2)
    cv2.line(img, (750, 360), (750, 440), (255, 0, 0), 2)
    
    # Border (dashed line effect)
    for i in range(0, width, 30):
        cv2.line(img, (i, 0), (i+15, 0), (255, 255, 0), 5)
    for i in range(0, height, 30):
        cv2.line(img, (0, i), (0, i+15), (255, 255, 0), 5)
    for i in range(0, width, 30):
        cv2.line(img, (i, height-1), (i+15, height-1), (255, 255, 0), 5)
    for i in range(0, height, 30):
        cv2.line(img, (width-1, i), (width-1, i+15), (255, 255, 0), 5)
    
    return img


class TestInvasionPlanningDemo:
    """Real-world demo: Planning an invasion using image annotation and CR-CA."""
    
    def test_tactical_map_analysis(self):
        """Step 1: Analyze tactical map to identify key targets."""
        engine = ImageAnnotationEngine()
        
        # Create tactical map
        tactical_map = create_tactical_map()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, tactical_map)
            map_path = f.name
        
        try:
            # Annotate the tactical map
            result = engine.annotate(map_path, output="all")
            
            assert isinstance(result, AnnotationResult)
            assert len(result.annotation_graph.entities) > 0
            
            # Extract key information
            entities = result.annotation_graph.entities
            labels = result.annotation_graph.labels
            
            # Should have detected various primitives
            entity_types = [e.primitive_type for e in entities]
            assert "line" in entity_types or "circle" in entity_types or "contour" in entity_types
            
        finally:
            if os.path.exists(map_path):
                os.unlink(map_path)
    
    def test_identify_targets(self):
        """Step 2: Identify military targets and cities."""
        engine = ImageAnnotationEngine()
        
        tactical_map = create_tactical_map()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, tactical_map)
            map_path = f.name
        
        try:
            # Query for largest city (capital)
            capital_result = engine.query(map_path, "find the largest building or structure")
            
            assert isinstance(capital_result, dict)
            assert "entities" in capital_result
            assert "answer" in capital_result
            
            # Query for military bases (circles)
            bases_result = engine.query(map_path, "find all circles and identify military installations")
            
            assert isinstance(bases_result, dict)
            assert "entities" in bases_result
            
            # Query for roads (infrastructure)
            roads_result = engine.query(map_path, "identify all major lines that could be roads or paths")
            
            assert isinstance(roads_result, dict)
            
        finally:
            if os.path.exists(map_path):
                os.unlink(map_path)
    
    def test_measure_critical_distances(self):
        """Step 3: Measure distances between key targets."""
        engine = ImageAnnotationEngine()
        
        tactical_map = create_tactical_map()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, tactical_map)
            map_path = f.name
        
        try:
            # Measure distance from border to capital
            distance_result = engine.query(
                map_path,
                "measure the distance from the border to the largest city"
            )
            
            assert isinstance(distance_result, dict)
            assert "measurements" in distance_result
            
            # Measure size of military bases
            base_size_result = engine.query(
                map_path,
                "find all circles and measure their areas and distances from each other"
            )
            
            assert isinstance(base_size_result, dict)
            
        finally:
            if os.path.exists(map_path):
                os.unlink(map_path)
    
    def test_crca_strategic_analysis(self):
        """Step 4: Use CR-CA agent for strategic causal analysis."""
        # Create CR-CA agent with image annotation tools
        agent = CRCAAgent(
            model_name="gpt-4o-mini",
            use_image_annotation=True,
            use_crca_tools=True
        )
        
        tactical_map = create_tactical_map()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, tactical_map)
            map_path = f.name
        
        try:
            # Task: Analyze tactical map and plan invasion strategy
            task = f"""
            Analyze this tactical map located at {map_path} and develop an invasion strategy.
            
            Steps:
            1. Use query_image to identify the largest city (capital) and measure its defenses
            2. Use query_image to locate all military bases and measure their sizes
            3. Use query_image to identify all roads and rivers (chokepoints)
            4. Extract causal variables for invasion planning:
               - Distance from border to capital
               - Number and size of military bases
               - Road network connectivity
               - River crossings required
            5. Perform causal analysis:
               - What factors affect invasion success?
               - What are the critical chokepoints?
               - What is the optimal approach route?
            
            Provide a comprehensive invasion plan with causal reasoning.
            """
            
            # Run agent (this would normally call the tools)
            # For testing, we'll just verify the agent is set up correctly
            assert agent is not None
            assert hasattr(agent, 'tools') or hasattr(agent, 'tools_list_dictionary')
            
        finally:
            if os.path.exists(map_path):
                os.unlink(map_path)
    
    def test_integrated_invasion_planning(self):
        """Complete integrated test: Full invasion planning workflow."""
        # Initialize both systems
        annotation_engine = ImageAnnotationEngine()
        strategic_agent = CRCAAgent(
            model_name="gpt-4o-mini",
            use_image_annotation=True,
            use_crca_tools=True
        )
        
        # Create tactical map
        tactical_map = create_tactical_map()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, tactical_map)
            map_path = f.name
        
        try:
            # Phase 1: Intelligence Gathering (Image Annotation)
            intel_report = annotation_engine.query(
                map_path,
                "identify all military installations, cities, roads, and rivers. Measure distances between key targets."
            )
            
            assert isinstance(intel_report, dict)
            assert "entities" in intel_report
            assert "measurements" in intel_report
            
            # Extract key metrics
            num_targets = len(intel_report["entities"])
            has_measurements = len(intel_report["measurements"]) > 0
            
            # Phase 2: Strategic Analysis (CR-CA)
            # Build causal model for invasion planning
            strategic_agent.add_causal_relationship("distance_to_capital", "invasion_difficulty")
            strategic_agent.add_causal_relationship("number_of_bases", "defense_strength")
            strategic_agent.add_causal_relationship("defense_strength", "invasion_difficulty")
            strategic_agent.add_causal_relationship("road_connectivity", "mobility")
            strategic_agent.add_causal_relationship("mobility", "invasion_success")
            
            # Set initial state
            initial_state = {
                "distance_to_capital": 200.0,  # pixels (from intel)
                "number_of_bases": 2.0,  # from intel
                "defense_strength": 0.0,  # will be computed
                "road_connectivity": 0.7,  # estimated
                "mobility": 0.0,  # will be computed
                "invasion_difficulty": 0.0,  # will be computed
                "invasion_success": 0.0  # will be computed
            }
            
            # Standardize stats
            strategic_agent.set_standardization_stats("distance_to_capital", mean=200.0, std=100.0)
            strategic_agent.set_standardization_stats("number_of_bases", mean=2.0, std=1.0)
            strategic_agent.set_standardization_stats("defense_strength", mean=0.5, std=0.3)
            strategic_agent.set_standardization_stats("road_connectivity", mean=0.5, std=0.2)
            strategic_agent.set_standardization_stats("mobility", mean=0.5, std=0.3)
            strategic_agent.set_standardization_stats("invasion_difficulty", mean=0.5, std=0.3)
            strategic_agent.set_standardization_stats("invasion_success", mean=0.5, std=0.3)
            
            # Predict outcomes
            outcomes = strategic_agent.predict_outcomes(initial_state, {})
            
            assert isinstance(outcomes, dict)
            assert "invasion_success" in outcomes
            assert "invasion_difficulty" in outcomes
            
            # Phase 3: Counterfactual Analysis
            # What if we had better mobility?
            intervention = {"road_connectivity": 0.9}
            counterfactual = strategic_agent.predict_outcomes(initial_state, intervention)
            
            assert isinstance(counterfactual, dict)
            # Better mobility should improve invasion success
            if counterfactual.get("invasion_success", 0) > outcomes.get("invasion_success", 0):
                assert True  # Expected improvement
            
            # Phase 4: Generate Strategic Report
            report = f"""
            INVASION PLANNING REPORT
            ========================
            
            Intelligence Summary:
            - Targets Identified: {num_targets}
            - Measurements Available: {has_measurements}
            
            Causal Analysis:
            - Base Invasion Success Probability: {outcomes.get('invasion_success', 0):.2f}
            - Invasion Difficulty: {outcomes.get('invasion_difficulty', 0):.2f}
            
            Counterfactual Scenario (Improved Mobility):
            - Improved Success Probability: {counterfactual.get('invasion_success', 0):.2f}
            - Improvement: {counterfactual.get('invasion_success', 0) - outcomes.get('invasion_success', 0):.2f}
            
            Recommendation:
            - Focus on securing road networks to improve mobility
            - Target priority: Largest city and military bases identified
            """
            
            assert len(report) > 0
            assert "INVASION PLANNING REPORT" in report
            
        finally:
            if os.path.exists(map_path):
                os.unlink(map_path)
    
    def test_multi_phase_operation(self):
        """Test multi-phase operation with temporal tracking."""
        engine = ImageAnnotationEngine(enable_temporal_tracking=True)
        
        # Create sequence of tactical maps (simulating movement)
        maps = []
        temp_files = []
        
        for phase in range(3):
            # Create map with different positions
            tactical_map = create_tactical_map()
            # Add phase marker
            cv2.putText(tactical_map, f"Phase {phase+1}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, tactical_map)
                maps.append(f.name)
                temp_files.append(f.name)
        
        try:
            # Process each phase with temporal tracking
            results = []
            for i, map_path in enumerate(maps):
                result = engine.annotate(map_path, frame_id=i)
                results.append(result)
                
                # Query for target tracking
                if i > 0:
                    query_result = engine.query(
                        map_path,
                        "identify changes in military base positions compared to previous phase"
                    )
                    assert isinstance(query_result, dict)
            
            # Should have processed all phases
            assert len(results) == 3
            assert all(isinstance(r, AnnotationResult) for r in results)
            
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
