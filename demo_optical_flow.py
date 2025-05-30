#!/usr/bin/env python3
"""
Demo script for testing optical flow analysis with LLaVA in ME-VQA.
"""
import argparse
import os
import json
from models.vlm_connector import VLMConnector
from utils.data_utils import load_jsonl, get_frame_differences_with_optical_flow


def test_optical_flow_analysis(connector, image_dir, question):
    """Test optical flow analysis on a single image directory."""
    print(f"\nTesting Optical Flow Analysis")
    print(f"Image directory: {image_dir}")
    print(f"Question: {question}")
    
    # Get frame data with optical flow
    frame_data = get_frame_differences_with_optical_flow(image_dir)
    if frame_data is None:
        print("Error: Could not compute optical flow")
        return {}
    
    results = {}
    
    # Test standard method
    try:
        answer = connector.answer_question(image_dir, question, use_middle_frame=True)
        print(f"Standard method: {answer}")
        results["standard"] = answer
    except Exception as e:
        print(f"Error with standard method: {e}")
        results["standard"] = f"Error: {e}"
    
    # Test optical flow color wheel
    try:
        answer = connector.answer_question_with_optical_flow(
            image_dir, question, flow_type="first_to_middle", visualization_type="color_wheel"
        )
        print(f"Optical flow (color wheel): {answer}")
        results["optical_flow_color"] = answer
    except Exception as e:
        print(f"Error with optical flow color: {e}")
        results["optical_flow_color"] = f"Error: {e}"
    
    # Test optical flow magnitude
    try:
        answer = connector.answer_question_with_optical_flow(
            image_dir, question, flow_type="first_to_middle", visualization_type="magnitude"
        )
        print(f"Optical flow (magnitude): {answer}")
        results["optical_flow_magnitude"] = answer
    except Exception as e:
        print(f"Error with optical flow magnitude: {e}")
        results["optical_flow_magnitude"] = f"Error: {e}"
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Demo for Optical Flow Analysis")
    parser.add_argument("--model_name", type=str, default="llava-1.5-7b")
    parser.add_argument("--image_dir", type=str, help="Path to image directory")
    parser.add_argument("--question", type=str, default="What is the coarse expression class?")
    
    args = parser.parse_args()
    
    if not args.image_dir:
        print("Please provide --image_dir")
        return
    
    # Initialize VLM connector
    print(f"Initializing {args.model_name}...")
    connector = VLMConnector(args.model_name)
    
    # Test optical flow analysis
    results = test_optical_flow_analysis(connector, args.image_dir, args.question)
    
    print(f"\nResults summary:")
    for method, answer in results.items():
        print(f"  {method}: {answer}")


if __name__ == "__main__":
    main() 
"""
Demo script for testing optical flow analysis with LLaVA in ME-VQA.
This script demonstrates how LLaVA can interpret optical flow visualizations.
"""
import argparse
import os
import json
from models.vlm_connector import VLMConnector
from utils.data_utils import load_jsonl, get_frame_differences_with_optical_flow
import matplotlib.pyplot as plt
from PIL import Image


def test_optical_flow_analysis(connector, image_dir, question, save_visualizations=True):
    """
    Test optical flow analysis on a single image directory.
    
    Args:
        connector: VLM connector instance
        image_dir: Directory containing image sequence
        question: Question to ask
        save_visualizations: Whether to save visualization images
    """
    print(f"\n{'='*60}")
    print(f"Testing Optical Flow Analysis")
    print(f"{'='*60}")
    print(f"Image directory: {image_dir}")
    print(f"Question: {question}")
    
    # Get frame data with optical flow
    frame_data = get_frame_differences_with_optical_flow(image_dir)
    if frame_data is None:
        print("Error: Could not compute optical flow")
        return
    
    # Test different visualization methods
    methods = [
        ("standard", "use_middle_frame", None, None),
        ("frame_difference", "first_to_middle", "first_to_middle", None),
        ("optical_flow", "first_to_middle_color", "first_to_middle", "color_wheel"),
        ("optical_flow", "first_to_middle_magnitude", "first_to_middle", "magnitude"),
        ("optical_flow", "middle_to_last_color", "middle_to_last", "color_wheel"),
        ("optical_flow", "middle_to_last_magnitude", "middle_to_last", "magnitude")
    ]
    
    results = {}
    
    for method_name, method_type, flow_type, viz_type in methods:
        print(f"\n{'-'*40}")
        print(f"Method: {method_name}")
        if flow_type:
            print(f"Flow type: {flow_type}")
        if viz_type:
            print(f"Visualization: {viz_type}")
        
        try:
            if method_name == "standard":
                answer = connector.answer_question(image_dir, question, use_middle_frame=True)
            elif method_name == "frame_difference":
                answer = connector.answer_question_with_frame_difference(
                    image_dir, question, difference_type=flow_type
                )
            elif method_name == "optical_flow":
                answer = connector.answer_question_with_optical_flow(
                    image_dir, question, flow_type=flow_type, visualization_type=viz_type
                )
            
            print(f"Answer: {answer}")
            results[f"{method_name}_{method_type}"] = answer
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[f"{method_name}_{method_type}"] = f"Error: {e}"
    
    # Save visualizations if requested
    if save_visualizations and frame_data:
        save_optical_flow_visualizations(frame_data, image_dir, question)
    
    return results


def save_optical_flow_visualizations(frame_data, image_dir, question):
    """Save comprehensive visualizations including optical flow."""
    try:
        # Create visualization directory
        vis_dir = os.path.join("visualizations", "optical_flow", os.path.basename(image_dir))
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load original images
        first_img = Image.open(frame_data['first_frame_path'])
        middle_img = Image.open(frame_data['middle_frame_path'])
        last_img = Image.open(frame_data['last_frame_path'])
        
        # Create comprehensive figure
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f"Comprehensive Analysis for {os.path.basename(image_dir)}\nQuestion: {question}", fontsize=16)
        
        # Row 1: Original frames
        axes[0, 0].imshow(first_img)
        axes[0, 0].set_title("First Frame")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(middle_img)
        axes[0, 1].set_title("Middle Frame")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(last_img)
        axes[0, 2].set_title("Last Frame")
        axes[0, 2].axis('off')
        
        # Hide unused subplot
        axes[0, 3].axis('off')
        
        # Row 2: Frame differences
        if frame_data['first_to_middle_diff']:
            axes[1, 0].imshow(frame_data['first_to_middle_diff'])
            axes[1, 0].set_title("Frame Diff: First→Middle")
            axes[1, 0].axis('off')
        
        if frame_data['middle_to_last_diff']:
            axes[1, 1].imshow(frame_data['middle_to_last_diff'])
            axes[1, 1].set_title("Frame Diff: Middle→Last")
            axes[1, 1].axis('off')
        
        # Hide unused subplots
        axes[1, 2].axis('off')
        axes[1, 3].axis('off')
        
        # Row 3: Optical flow visualizations
        if frame_data['first_to_middle_flow_color']:
            axes[2, 0].imshow(frame_data['first_to_middle_flow_color'])
            axes[2, 0].set_title("Optical Flow: First→Middle (Color Wheel)")
            axes[2, 0].axis('off')
        
        if frame_data['first_to_middle_flow_magnitude']:
            axes[2, 1].imshow(frame_data['first_to_middle_flow_magnitude'])
            axes[2, 1].set_title("Optical Flow: First→Middle (Magnitude)")
            axes[2, 1].axis('off')
        
        if frame_data['middle_to_last_flow_color']:
            axes[2, 2].imshow(frame_data['middle_to_last_flow_color'])
            axes[2, 2].set_title("Optical Flow: Middle→Last (Color Wheel)")
            axes[2, 2].axis('off')
        
        if frame_data['middle_to_last_flow_magnitude']:
            axes[2, 3].imshow(frame_data['middle_to_last_flow_magnitude'])
            axes[2, 3].set_title("Optical Flow: Middle→Last (Magnitude)")
            axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        save_path = os.path.join(vis_dir, "comprehensive_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive visualization saved to: {save_path}")
        
        # Also save individual optical flow images for closer inspection
        individual_dir = os.path.join(vis_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        if frame_data['first_to_middle_flow_color']:
            frame_data['first_to_middle_flow_color'].save(
                os.path.join(individual_dir, "first_to_middle_flow_color.png")
            )
        
        if frame_data['first_to_middle_flow_magnitude']:
            frame_data['first_to_middle_flow_magnitude'].save(
                os.path.join(individual_dir, "first_to_middle_flow_magnitude.png")
            )
        
        print(f"Individual optical flow images saved to: {individual_dir}")
        
    except Exception as e:
        print(f"Error saving optical flow visualizations: {e}")


def run_optical_flow_comparison(connector, jsonl_file, max_samples=3):
    """Run comparison between standard, frame difference, and optical flow methods."""
    print(f"\n{'='*60}")
    print(f"Running Optical Flow Comparison")
    print(f"{'='*60}")
    
    data = load_jsonl(jsonl_file)
    if max_samples > 0:
        data = data[:max_samples]
    
    all_results = []
    
    for i, item in enumerate(data):
        dataset_name = item['dataset']
        image_id = item['image_id']
        question = item['question']
        ground_truth = item['answer']
        
        # Construct image path
        base_path = os.path.join("/home/tpei0009/MMNet", dataset_name, f"{image_id[0:3]}", f"{image_id}")
        
        if not os.path.isdir(base_path):
            print(f"Skipping {image_id}: not a directory")
            continue
        
        print(f"\nSample {i+1}: {image_id}")
        print(f"Ground truth: {ground_truth}")
        
        # Test methods
        results = test_optical_flow_analysis(connector, base_path, question, save_visualizations=False)
        results.update({
            "id": item['id'],
            "image_id": image_id,
            "question": question,
            "ground_truth": ground_truth
        })
        
        all_results.append(results)
    
    # Save comparison results
    with open("optical_flow_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nOptical flow comparison results saved to: optical_flow_comparison.json")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Demo for Optical Flow Analysis with LLaVA in ME-VQA")
    parser.add_argument("--model_name", type=str, default="llava-1.5-7b",
                        help="Name of the VLM model to use")
    parser.add_argument("--image_dir", type=str,
                        help="Path to image directory for single test")
    parser.add_argument("--question", type=str, default="What is the coarse expression class?",
                        help="Question to ask about the image")
    parser.add_argument("--jsonl_file", type=str, default="me_vqa_samm_casme2_smic.jsonl",
                        help="Path to JSONL file for batch comparison")
    parser.add_argument("--max_samples", type=int, default=2,
                        help="Maximum number of samples for batch comparison")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save optical flow visualizations")
    
    args = parser.parse_args()
    
    # Initialize VLM connector
    print(f"Initializing {args.model_name}...")
    connector = VLMConnector(args.model_name)
    
    # Single image directory test
    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"Error: {args.image_dir} is not a directory")
            return
        
        results = test_optical_flow_analysis(
            connector, 
            args.image_dir, 
            args.question,
            args.save_visualizations
        )
        
        print(f"\nResults summary:")
        for method, answer in results.items():
            print(f"  {method}: {answer}")
    
    # Batch comparison
    elif args.jsonl_file:
        if not os.path.exists(args.jsonl_file):
            print(f"Error: {args.jsonl_file} not found")
            return
        
        results = run_optical_flow_comparison(connector, args.jsonl_file, args.max_samples)
        
        print(f"\nProcessed {len(results)} samples")
        print("Check 'optical_flow_comparison.json' for detailed results")
    
    else:
        print("Please provide either --image_dir for single test or --jsonl_file for batch comparison")


if __name__ == "__main__":
    main() 
 