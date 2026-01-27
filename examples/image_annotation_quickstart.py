"""
Quick Start Guide for Image Annotation

This example demonstrates the most common use cases for the image annotation system.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from image_annotation import ImageAnnotationEngine
    from CRCA import CRCAAgent
    IMAGE_ANNOTATION_AVAILABLE = True
except ImportError as e:
    print(f"Image annotation not available: {e}")
    print("Make sure all dependencies are installed:")
    print("  pip install opencv-python numpy pillow loguru rustworkx")
    IMAGE_ANNOTATION_AVAILABLE = False


def example_1_basic_annotation():
    """Example 1: Basic image annotation."""
    print("\n=== Example 1: Basic Image Annotation ===")
    
    if not IMAGE_ANNOTATION_AVAILABLE:
        print("Skipping - image annotation not available")
        return
    
    # Initialize the engine
    engine = ImageAnnotationEngine()
    
    # Example: Annotate an image file
    # Replace with your image path
    image_path = "path/to/your/image.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update image_path with a valid image file")
        return
    
    # Annotate the image
    result = engine.annotate(image_path, output="all")
    
    # Access results
    print(f"Found {len(result.annotation_graph.entities)} geometric primitives")
    print(f"Generated {len(result.annotation_graph.labels)} semantic labels")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    
    # Print some labels
    print("\nLabels:")
    for label in result.annotation_graph.labels[:5]:  # First 5 labels
        print(f"  - {label.label} (uncertainty: {label.uncertainty:.2f})")
    
    # Save overlay image
    if result.overlay_image:
        import cv2
        import numpy as np
        overlay = cv2.imdecode(
            np.frombuffer(result.overlay_image, np.uint8),
            cv2.IMREAD_COLOR
        )
        cv2.imwrite("annotated_output.png", overlay)
        print("\nSaved annotated image to: annotated_output.png")


def example_2_query_image():
    """Example 2: Query an image with natural language."""
    print("\n=== Example 2: Query Image ===")
    
    if not IMAGE_ANNOTATION_AVAILABLE:
        print("Skipping - image annotation not available")
        return
    
    engine = ImageAnnotationEngine()
    
    image_path = "path/to/your/image.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update image_path with a valid image file")
        return
    
    # Query 1: Find specific objects
    print("\nQuery 1: Find all circles")
    result = engine.query(image_path, "find all circles")
    print(f"Answer: {result['answer']}")
    print(f"Found {len(result['entities'])} circles")
    
    # Query 2: Measure something
    print("\nQuery 2: Measure distances")
    result = engine.query(image_path, "measure the distance from the border to the largest structure")
    print(f"Answer: {result['answer']}")
    if result.get('measurements'):
        print(f"Measurements: {result['measurements']}")
    
    # Query 3: Identify objects
    print("\nQuery 3: Identify components")
    result = engine.query(image_path, "identify all components in this image")
    print(f"Answer: {result['answer']}")


def example_3_different_output_formats():
    """Example 3: Using different output formats."""
    print("\n=== Example 3: Different Output Formats ===")
    
    if not IMAGE_ANNOTATION_AVAILABLE:
        print("Skipping - image annotation not available")
        return
    
    engine = ImageAnnotationEngine()
    
    image_path = "path/to/your/image.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update image_path with a valid image file")
        return
    
    # Output format: overlay (numpy array)
    print("\n1. Getting overlay image (numpy array)...")
    overlay = engine.annotate(image_path, output="overlay")
    print(f"   Overlay shape: {overlay.shape}")
    import cv2
    cv2.imwrite("overlay_output.png", overlay)
    print("   Saved to: overlay_output.png")
    
    # Output format: JSON
    print("\n2. Getting JSON data...")
    json_data = engine.annotate(image_path, output="json")
    print(f"   JSON keys: {list(json_data.keys())}")
    print(f"   Number of entities: {len(json_data.get('entities', []))}")
    
    # Output format: report
    print("\n3. Getting formal report...")
    report = engine.annotate(image_path, output="report")
    print(f"   Report length: {len(report)} characters")
    print(f"   First 200 chars: {report[:200]}...")
    
    # Output format: all (complete result)
    print("\n4. Getting complete result...")
    result = engine.annotate(image_path, output="all")
    print(f"   Type: {type(result)}")
    print(f"   Has annotation_graph: {hasattr(result, 'annotation_graph')}")
    print(f"   Has overlay_image: {hasattr(result, 'overlay_image')}")
    print(f"   Has formal_report: {hasattr(result, 'formal_report')}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple images."""
    print("\n=== Example 4: Batch Processing ===")
    
    if not IMAGE_ANNOTATION_AVAILABLE:
        print("Skipping - image annotation not available")
        return
    
    engine = ImageAnnotationEngine(
        cache_enabled=True,  # Enable caching for faster processing
        show_progress=True   # Show progress bar (requires tqdm)
    )
    
    # List of image paths
    image_paths = [
        "path/to/image1.png",
        "path/to/image2.png",
        "path/to/image3.png"
    ]
    
    # Filter to only existing files
    existing_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not existing_paths:
        print("No image files found. Please update image_paths with valid files")
        return
    
    print(f"Processing {len(existing_paths)} images...")
    
    # Process all images
    results = engine.annotate(existing_paths, output="all")
    
    # Analyze results
    print(f"\nProcessed {len(results)} images:")
    for i, result in enumerate(results):
        print(f"  Image {i+1}: {len(result.annotation_graph.entities)} entities, "
              f"{len(result.annotation_graph.labels)} labels, "
              f"{result.processing_time:.2f}s")


def example_5_with_crca_agent():
    """Example 5: Using image annotation with CRCAAgent."""
    print("\n=== Example 5: Integration with CRCAAgent ===")
    
    if not IMAGE_ANNOTATION_AVAILABLE:
        print("Skipping - image annotation not available")
        return
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set it to use CRCAAgent.")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    # Create agent with image annotation enabled
    agent = CRCAAgent(
        model_name="gpt-4o-mini",
        use_image_annotation=True,  # Enable image annotation tools
        use_crca_tools=True
    )
    
    image_path = "path/to/your/image.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update image_path with a valid image file")
        return
    
    # Task that uses image annotation
    task = f"""
    Analyze the image at {image_path}:
    1. Use query_image to identify all objects in the image
    2. Use query_image to find the largest object
    3. Use query_image to measure distances between key objects
    4. Summarize your findings
    """
    
    print("Running agent with image annotation tools...")
    print("(This will make API calls to OpenAI)")
    
    # Uncomment to actually run (requires API key and credits)
    # response = agent.run(task)
    # print(f"\nAgent response:\n{response}")
    
    print("\nNote: Uncomment the code above to actually run the agent")


def example_6_different_input_types():
    """Example 6: Using different input types."""
    print("\n=== Example 6: Different Input Types ===")
    
    if not IMAGE_ANNOTATION_AVAILABLE:
        print("Skipping - image annotation not available")
        return
    
    engine = ImageAnnotationEngine()
    
    # 1. File path (string)
    image_path = "path/to/your/image.png"
    if os.path.exists(image_path):
        print("1. Annotating from file path (string)...")
        result = engine.annotate(image_path, output="json")
        print(f"   Success: {len(result.get('entities', []))} entities")
    
    # 2. Path object
    image_path_obj = Path("path/to/your/image.png")
    if image_path_obj.exists():
        print("\n2. Annotating from Path object...")
        result = engine.annotate(image_path_obj, output="json")
        print(f"   Success: {len(result.get('entities', []))} entities")
    
    # 3. NumPy array
    try:
        import cv2
        import numpy as np
        if os.path.exists(image_path):
            print("\n3. Annotating from NumPy array...")
            img = cv2.imread(image_path)
            result = engine.annotate(img, output="json")
            print(f"   Success: {len(result.get('entities', []))} entities")
    except ImportError:
        print("\n3. Skipping NumPy array example (cv2 not available)")
    
    # 4. PIL Image
    try:
        from PIL import Image as PILImage
        if os.path.exists(image_path):
            print("\n4. Annotating from PIL Image...")
            img = PILImage.open(image_path)
            result = engine.annotate(img, output="json")
            print(f"   Success: {len(result.get('entities', []))} entities")
    except ImportError:
        print("\n4. Skipping PIL Image example (PIL not available)")
    
    # 5. URL (if requests is available)
    try:
        import requests
        print("\n5. Annotating from URL...")
        url = "https://example.com/image.png"  # Replace with actual URL
        # Uncomment to test:
        # result = engine.annotate(url, output="json")
        # print(f"   Success: {len(result.get('entities', []))} entities")
        print("   (Skipped - update URL to test)")
    except ImportError:
        print("\n5. Skipping URL example (requests not available)")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Image Annotation Quick Start Examples")
    print("=" * 60)
    
    if not IMAGE_ANNOTATION_AVAILABLE:
        print("\nImage annotation is not available.")
        print("Please install required dependencies:")
        print("  pip install opencv-python numpy pillow loguru rustworkx")
        return
    
    # Run examples
    example_1_basic_annotation()
    example_2_query_image()
    example_3_different_output_formats()
    example_4_batch_processing()
    example_5_with_crca_agent()
    example_6_different_input_types()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - docs/IMAGE_ANNOTATION_USAGE.md")
    print("  - tests/test_image_annotation_*.py")


if __name__ == "__main__":
    main()
