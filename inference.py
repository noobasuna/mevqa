#!/usr/bin/env python3
import argparse
import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from models.vlm_connector import VLMConnector
from utils.data_utils import load_jsonl, get_middle_frame


def parse_args():
    parser = argparse.ArgumentParser(description="ME-VQA Inference with VLMs")
    parser.add_argument("--model_name", type=str, default="llava-1.5-7b",
                        help="Name of the VLM model to use")
    parser.add_argument("--image_path", type=str,
                        help="Path to the image file or directory")
    parser.add_argument("--question", type=str,
                        help="Question to ask about the image")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing dataset images")
    parser.add_argument("--jsonl_file", type=str,
                        help="Path to JSONL file for batch inference")
    parser.add_argument("--output_file", type=str, default="results.json",
                        help="Path to save inference results")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Maximum number of samples to process for batch inference")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--use_middle_frame", action="store_true",
                        help="Use middle frame from image folders")
    return parser.parse_args()


def visualize_result(image_path, question, answer):
    """Visualize the result with matplotlib."""
    image = Image.open(image_path).convert("RGB")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Q: {question}\nA: {answer}", fontsize=14)
    plt.tight_layout()
    plt.show()


def single_image_inference(connector, image_path, question, use_middle_frame=False):
    """Run inference on a single image or directory of images."""
    print(f"\nProcessing image: {image_path}")
    print(f"Question: {question}")
    
    # For visualization, we need the actual image file if it's a directory
    display_path = image_path
    if os.path.isdir(image_path) and use_middle_frame:
        middle_frame_path = get_middle_frame(image_path)
        if middle_frame_path:
            display_path = middle_frame_path
    
    # Pass the original path to the connector - it will handle the middle frame logic
    answer = connector.answer_question(image_path, question, use_middle_frame=use_middle_frame)
    print(f"Answer: {answer}")
    
    return {
        "image_path": image_path,
        "display_path": display_path,
        "question": question,
        "answer": answer
    }


def batch_inference(connector, jsonl_file, data_dir, max_samples=10, use_middle_frame=False):
    """Run inference on multiple samples from a JSONL file."""
    data = load_jsonl(jsonl_file)
    
    if max_samples > 0:
        data = data[:max_samples]
    
    results = []
    
    for i, item in enumerate(data):
        dataset_name = item['dataset']
        image_id = item['image_id']
        question = item['question']
        ground_truth = item['answer']
        
        # Construct base path
        base_path = os.path.join(data_dir, dataset_name, f"{image_id}")
        
        # Handle different path possibilities based on use_middle_frame
        if use_middle_frame:
            # Check if it's a directory first
            if os.path.isdir(base_path):
                image_path = base_path  # Will get middle frame later
                print(f"Using image directory: {image_path}")
            else:
                # Try with .jpg extension
                image_path = f"{base_path}.jpg"
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
        else:
            # Default behavior - expect a .jpg file
            image_path = f"{base_path}.jpg"
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                continue
        
        print(f"\nProcessing sample {i+1}/{len(data)}")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        # For visualization/display, get actual file path if it's a directory
        display_path = image_path
        if os.path.isdir(image_path) and use_middle_frame:
            middle_frame_path = get_middle_frame(image_path)
            if middle_frame_path:
                display_path = middle_frame_path
        
        # Run inference
        answer = connector.answer_question(image_path, question, use_middle_frame=use_middle_frame)
        print(f"Predicted: {answer}")
        
        # Save result
        results.append({
            "id": item['id'],
            "dataset": dataset_name,
            "image_id": image_id,
            "image_path": image_path,
            "display_path": display_path,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": answer
        })
    
    return results


def main():
    args = parse_args()
    
    # List available models and exit
    if args.list_models:
        connector = VLMConnector()
        print("Available models:")
        for model in connector.available_models():
            print(f"  - {model}")
        return
    
    # Initialize VLM connector
    print(f"Initializing {args.model_name}...")
    connector = VLMConnector(args.model_name)
    
    # Single image inference
    if args.image_path and args.question:
        result = single_image_inference(connector, args.image_path, args.question, args.use_middle_frame)
        
        # Optionally visualize the result
        if os.environ.get("DISPLAY"):
            # Use display_path for visualization (the actual image file, not the directory)
            visualize_result(result["display_path"], args.question, result["answer"])
            
    # Batch inference from JSONL file
    elif args.jsonl_file:
        print(f"Running batch inference on {args.jsonl_file}")
        results = batch_inference(connector, args.jsonl_file, args.data_dir, args.max_samples, args.use_middle_frame)
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessed {len(results)} samples")
        print(f"Results saved to {args.output_file}")
    
    else:
        print("Error: Please provide either --image_path and --question for single inference, "
              "or --jsonl_file for batch inference")


if __name__ == "__main__":
    main() 