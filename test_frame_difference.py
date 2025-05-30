#!/usr/bin/env python3
"""
Simple test script to verify frame difference functionality.
"""
import os
import sys
from utils.data_utils import get_frame_differences, get_first_frame, get_middle_frame, get_last_frame

def test_frame_utils():
    """Test the frame utility functions."""
    print("Testing Frame Utility Functions")
    print("=" * 50)
    
    # Test with a sample from the JSONL data
    sample_dirs = [
        "/home/tpei0009/MMNet/samm/006/006_1_2",
        "/home/tpei0009/MMNet/samm/006/006_1_3",
        "/home/tpei0009/MMNet/casme2/s15/15_0101",
    ]
    
    for image_dir in sample_dirs:
        print(f"\nTesting directory: {image_dir}")
        
        if not os.path.isdir(image_dir):
            print(f"  Directory not found, skipping...")
            continue
        
        # Test individual frame functions
        first_frame = get_first_frame(image_dir)
        middle_frame = get_middle_frame(image_dir)
        last_frame = get_last_frame(image_dir)
        
        print(f"  First frame: {first_frame}")
        print(f"  Middle frame: {middle_frame}")
        print(f"  Last frame: {last_frame}")
        
        # Test frame differences
        frame_data = get_frame_differences(image_dir)
        if frame_data:
            print(f"  Frame differences computed successfully!")
            print(f"  First to middle diff: {frame_data['first_to_middle_diff'] is not None}")
            print(f"  Middle to last diff: {frame_data['middle_to_last_diff'] is not None}")
        else:
            print(f"  Error computing frame differences")


def test_vlm_connector():
    """Test the VLM connector with frame differences."""
    print("\n\nTesting VLM Connector")
    print("=" * 50)
    
    try:
        from models.vlm_connector import VLMConnector
        
        # Try to find a sample directory
        sample_dir = "/home/tpei0009/MMNet/samm/006/006_1_2"
        if not os.path.isdir(sample_dir):
            print("Sample directory not found, skipping VLM test")
            return
        
        print(f"Testing with directory: {sample_dir}")
        
        # Initialize connector (this might take some time)
        print("Initializing VLM connector (this may take a moment)...")
        connector = VLMConnector("llava-1.5-7b")
        
        question = "What is the coarse expression class?"
        
        # Test standard method
        print("\nTesting standard method...")
        try:
            answer_standard = connector.answer_question(sample_dir, question, use_middle_frame=True)
            print(f"Standard answer: {answer_standard}")
        except Exception as e:
            print(f"Error with standard method: {e}")
        
        # Test frame difference method
        print("\nTesting frame difference method...")
        try:
            answer_diff = connector.answer_question_with_frame_difference(
                sample_dir, question, difference_type="first_to_middle"
            )
            print(f"Frame difference answer: {answer_diff}")
        except Exception as e:
            print(f"Error with frame difference method: {e}")
            
    except Exception as e:
        print(f"Error initializing VLM connector: {e}")
        print("This is expected if you don't have the model dependencies installed")


def main():
    """Run all tests."""
    print("ME-VQA Frame Difference Test Suite")
    print("=" * 60)
    
    # Test frame utilities (doesn't require GPU/models)
    test_frame_utils()
    
    # Test VLM connector (requires models, might fail in some environments)
    if "--skip-vlm" not in sys.argv:
        test_vlm_connector()
    else:
        print("\nSkipping VLM tests (--skip-vlm flag provided)")
    
    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    main() 