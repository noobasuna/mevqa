#!/usr/bin/env python3
import argparse
import os
import json
from models.vlm_connector import VLMConnector
from utils.visualization import display_image_with_answer, visualize_results_grid
from utils.data_utils import load_jsonl, analyze_dataset, get_middle_frame


def parse_args():
    parser = argparse.ArgumentParser(description="ME-VQA Demo Script")
    parser.add_argument("--model_name", type=str, default="llava-1.5-7b",
                        help="Name of the VLM model to use")
    parser.add_argument("--jsonl_file", type=str, default="me_vqa_samm_casme2_smic.jsonl",
                        help="Path to JSONL file")
    parser.add_argument("--data_dir", type=str, default="/home/tpei0009/MMNet",
                        help="Directory containing dataset images")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index to use for demo")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze the dataset")
    parser.add_argument("--use_middle_frame", action="store_true",
                        help="Use middle frame from image folders", default=True)
    parser.add_argument("--image_folder", type=str,
                        help="Directly use an image folder for inference")
    parser.add_argument("--question", type=str,
                        help="Question to ask when using image_folder")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List available models
    if args.list_models:
        connector = VLMConnector()
        print("Available models:")
        for model in connector.available_models():
            print(f"  - {model}")
        return
    
    # Analyze dataset
    if args.analyze:
        print(f"Analyzing dataset: {args.jsonl_file}")
        analyze_dataset(args.jsonl_file)
        return
    
    # Direct folder inference mode
    if args.image_folder and args.question:
        print(f"Using image folder: {args.image_folder}")
        
        # Check if folder exists
        if not os.path.exists(args.image_folder):
            print(f"Error: Folder does not exist: {args.image_folder}")
            return
            
        # Initialize model
        try:
            print(f"\nInitializing {args.model_name}...")
            connector = VLMConnector(args.model_name)
        except Exception as e:
            print(f"Error initializing model: {e}")
            return
            
        # Run inference
        try:
            print("\nRunning inference...")
            print(f"Question: {args.question}")
            
            # Get middle frame if specified
            if args.use_middle_frame and os.path.isdir(args.image_folder):
                print("Using middle frame from folder...")
                image_path = args.image_folder  # VLMConnector will handle getting the middle frame
            else:
                image_path = args.image_folder
                
            answer = connector.answer_question(image_path, args.question, use_middle_frame=args.use_middle_frame)
            print(f"Answer: {answer}")
            
            # If the path is a directory and middle frame is used, get the middle frame for visualization
            if os.path.isdir(image_path) and args.use_middle_frame:
                middle_frame_path = get_middle_frame(image_path)
                if middle_frame_path:
                    display_image_with_answer(middle_frame_path, args.question, answer)
            else:
                # Handle the case where it's a single image file
                display_image_with_answer(image_path, args.question, answer)
                
        except Exception as e:
            print(f"Error during inference: {e}")
            
        return
    
    # Load data from JSONL for regular mode
    try:
        data = load_jsonl(args.jsonl_file)
        print(f"Loaded {len(data)} samples from {args.jsonl_file}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure valid sample index
    if args.sample_idx >= len(data) or args.sample_idx < 0:
        print(f"Invalid sample index {args.sample_idx}. Using index 0.")
        args.sample_idx = 0

    # Get sample
    sample = data[args.sample_idx]
    dataset_name = sample['dataset']
    image_id = sample['image_id']
    question = sample['question']
    ground_truth = sample['answer']
    
    print("\nSelected sample:")
    print(f"Dataset: {dataset_name}")
    print(f"Image ID: {image_id}")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    
    # Construct image path
    base_path = os.path.join(args.data_dir, dataset_name,f"{image_id[0:3]}", f"{image_id}")
    
    # Handle different path possibilities based on use_middle_frame
    if args.use_middle_frame:
        # Check if it's a directory first
        if os.path.isdir(base_path):
            image_path = base_path  # VLMConnector will handle getting the middle frame
            print(f"Using image directory: {image_path}")
        else:
            # Try with .jpg extension
            image_path = f"{base_path}.jpg"
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                print("Please ensure you have downloaded the dataset images and placed them in the correct directory structure.")
                return
    else:
        # Default behavior - expect a .jpg file
        image_path = f"{base_path}.jpg"
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            print("Please ensure you have downloaded the dataset images and placed them in the correct directory structure.")
            return
        
    # Initialize model
    try:
        print(f"\nInitializing {args.model_name}...")
        connector = VLMConnector(args.model_name)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
        
    # Run inference
    try:
        print("\nRunning inference...")
        answer = connector.answer_question(image_path, question, use_middle_frame=args.use_middle_frame)
        print(f"\nQuestion: {question}")
        print(f"Model's answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        
        # For visualization, if it's a directory and we're using the middle frame, get the actual file path
        if os.path.isdir(image_path) and args.use_middle_frame:
            middle_frame_path = get_middle_frame(image_path)
            if middle_frame_path:
                display_image_with_answer(middle_frame_path, question, answer, ground_truth)
        else:
            # Normal case - direct image file
            display_image_with_answer(image_path, question, answer, ground_truth)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        
    print("\nDone!")


if __name__ == "__main__":
    main() 