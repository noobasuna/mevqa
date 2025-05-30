#!/usr/bin/env python3
"""
Demo script for testing frame difference analysis in ME-VQA.
This script demonstrates both the original middle frame method and the new frame difference methods.
"""
import argparse
import os
import json
from models.vlm_connector import VLMConnector
from utils.data_utils import load_jsonl, get_frame_differences
import matplotlib.pyplot as plt
from PIL import Image


def test_frame_difference_analysis(connector, image_dir, question, save_visualizations=True):
    """
    Test frame difference analysis on a single image directory.
    
    Args:
        connector: VLM connector instance
        image_dir: Directory containing image sequence
        question: Question to ask
        save_visualizations: Whether to save visualization images
    """
    print(f"\n{'='*60}")
    print(f"Testing Frame Difference Analysis")
    print(f"{'='*60}")
    print(f"Image directory: {image_dir}")
    print(f"Question: {question}")
    
    # Get frame data for visualization
    frame_data = get_frame_differences(image_dir)
    if frame_data is None:
        print("Error: Could not compute frame differences")
        return
    
    # Test different methods
    methods = [
        ("standard", "use_middle_frame", None),
        ("frame_difference", "first_to_middle", "first_to_middle"),
        ("frame_difference", "middle_to_last", "middle_to_last")
    ]
    
    results = {}
    
    for method_name, method_type, diff_type in methods:
        print(f"\n{'-'*40}")
        print(f"Method: {method_name}")
        if diff_type:
            print(f"Difference type: {diff_type}")
        
        try:
            if method_name == "standard":
                answer = connector.answer_question(image_dir, question, use_middle_frame=True)
            else:
                answer = connector.answer_question_with_frame_difference(image_dir, question, difference_type=diff_type)
            
            print(f"Answer: {answer}")
            results[f"{method_name}_{method_type}"] = answer
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[f"{method_name}_{method_type}"] = f"Error: {e}"
    
    # Save visualizations if requested
    if save_visualizations and frame_data:
        save_frame_visualizations(frame_data, image_dir, question)
    
    return results


def save_frame_visualizations(frame_data, image_dir, question):
    """Save visualizations of frames and differences."""
    try:
        # Create visualization directory
        vis_dir = os.path.join("visualizations", os.path.basename(image_dir))
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load images
        first_img = Image.open(frame_data['first_frame_path'])
        middle_img = Image.open(frame_data['middle_frame_path'])
        last_img = Image.open(frame_data['last_frame_path'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Frame Analysis for {os.path.basename(image_dir)}\nQuestion: {question}", fontsize=14)
        
        # Original frames
        axes[0, 0].imshow(first_img)
        axes[0, 0].set_title("First Frame")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(middle_img)
        axes[0, 1].set_title("Middle Frame")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(last_img)
        axes[0, 2].set_title("Last Frame")
        axes[0, 2].axis('off')
        
        # Difference images
        if frame_data['first_to_middle_diff']:
            axes[1, 0].imshow(frame_data['first_to_middle_diff'])
            axes[1, 0].set_title("First → Middle Difference")
            axes[1, 0].axis('off')
        
        if frame_data['middle_to_last_diff']:
            axes[1, 2].imshow(frame_data['middle_to_last_diff'])
            axes[1, 2].set_title("Middle → Last Difference")
            axes[1, 2].axis('off')
        
        # Hide the middle subplot in bottom row
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = os.path.join(vis_dir, "frame_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {save_path}")
        
    except Exception as e:
        print(f"Error saving visualizations: {e}")


def run_comparison_on_samples(connector, jsonl_file, max_samples=5):
    """Run comparison between methods on sample data."""
    print(f"\n{'='*60}")
    print(f"Running Comparison on Sample Data")
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
        results = test_frame_difference_analysis(connector, base_path, question, save_visualizations=False)
        results.update({
            "id": item['id'],
            "image_id": image_id,
            "question": question,
            "ground_truth": ground_truth
        })
        
        all_results.append(results)
    
    # Save comparison results
    with open("frame_difference_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComparison results saved to: frame_difference_comparison.json")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Demo for Frame Difference Analysis in ME-VQA")
    parser.add_argument("--model_name", type=str, default="llava-1.5-7b",
                        help="Name of the VLM model to use")
    parser.add_argument("--image_dir", type=str,
                        help="Path to image directory for single test")
    parser.add_argument("--question", type=str, default="What is the coarse expression class?",
                        help="Question to ask about the image")
    parser.add_argument("--jsonl_file", type=str, default="me_vqa_samm_casme2_smic.jsonl",
                        help="Path to JSONL file for batch comparison")
    parser.add_argument("--max_samples", type=int, default=3,
                        help="Maximum number of samples for batch comparison")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save frame visualizations")
    
    args = parser.parse_args()
    
    # Initialize VLM connector
    print(f"Initializing {args.model_name}...")
    connector = VLMConnector(args.model_name)
    
    # Single image directory test
    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"Error: {args.image_dir} is not a directory")
            return
        
        results = test_frame_difference_analysis(
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
        
        results = run_comparison_on_samples(connector, args.jsonl_file, args.max_samples)
        
        print(f"\nProcessed {len(results)} samples")
        print("Check 'frame_difference_comparison.json' for detailed results")
    
    else:
        print("Please provide either --image_dir for single test or --jsonl_file for batch comparison")


if __name__ == "__main__":
    main() 