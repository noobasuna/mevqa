#!/usr/bin/env python3
"""
Process ME-VQA JSONL file to generate answers using different VLM methods.
"""
import os
import sys
import json
import argparse
from tqdm import tqdm
from utils.data_utils import get_frame_differences, get_first_frame, get_middle_frame, get_last_frame

def construct_data_path(base_dir, filename):
    """
    Construct the data directory path from filename.
    
    Args:
        base_dir: Base directory (e.g., "samm_data")
        filename: Filename from JSONL (e.g., "004_6")
    
    Returns:
        Full path to the data directory
    """
    return os.path.join(base_dir, filename)

def process_jsonl_with_method(input_jsonl, output_jsonl, base_data_dir, method="middle_frame"):
    """
    Process JSONL file and generate answers using specified method.
    
    Args:
        input_jsonl: Path to input JSONL file
        output_jsonl: Path to output JSONL file
        base_data_dir: Base directory containing image data
        method: Method to use ("middle_frame", "first_frame", "last_frame", 
                "first_to_middle_diff", "middle_to_last_diff")
    """
    print(f"Processing with method: {method}")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_jsonl}")
    print("=" * 60)
    
    # Initialize VLM connector
    try:
        from models.vlm_connector import VLMConnector
        print("Initializing VLM connector...")
        connector = VLMConnector("llava-1.5-7b")
        print("VLM connector initialized successfully!")
    except Exception as e:
        print(f"Error initializing VLM connector: {e}")
        return
    
    # Read input JSONL
    with open(input_jsonl, 'r') as f:
        lines = f.readlines()
    
    results = []
    
    # Process each line
    for line in tqdm(lines, desc=f"Processing {method}"):
        try:
            data = json.loads(line.strip())
            
            # Construct data directory path
            data_dir = construct_data_path(base_data_dir, data['filename'])
            
            if not os.path.isdir(data_dir):
                print(f"Warning: Directory not found: {data_dir}")
                # Add entry with empty answer
                result = data.copy()
                result['answer'] = ""
                result['method'] = method
                result['error'] = "Directory not found"
                results.append(result)
                continue
            
            question = data['question']
            
            # Get answer based on method
            answer = ""
            try:
                if method == "middle_frame":
                    answer = connector.answer_question(data_dir, question, use_middle_frame=True)
                elif method == "first_frame":
                    answer = connector.answer_question(data_dir, question, use_first_frame=True)
                elif method == "last_frame":
                    answer = connector.answer_question(data_dir, question, use_last_frame=True)
                elif method == "first_to_middle_diff":
                    answer = connector.answer_question_with_frame_difference(
                        data_dir, question, difference_type="first_to_middle"
                    )
                elif method == "middle_to_last_diff":
                    answer = connector.answer_question_with_frame_difference(
                        data_dir, question, difference_type="middle_to_last"
                    )
                else:
                    answer = f"Unknown method: {method}"
                    
            except Exception as e:
                print(f"Error processing question for {data['id']}: {e}")
                answer = ""
            
            # Create result entry
            result = data.copy()
            result['answer'] = answer
            result['method'] = method
            results.append(result)
            
        except Exception as e:
            print(f"Error processing line: {e}")
            continue
    
    # Write results to output JSONL
    with open(output_jsonl, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Completed processing {len(results)} entries")
    print(f"Results saved to: {output_jsonl}")

def main():
    """Main function to process JSONL with different methods."""
    parser = argparse.ArgumentParser(description='Process ME-VQA JSONL file with different VLM methods')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--base_data_dir', required=True, help='Base directory containing image data (e.g., samm_data)')
    parser.add_argument('--output_dir', default='./results', help='Output directory for results')
    parser.add_argument('--methods', nargs='+', 
                       default=['middle_frame', 'first_frame', 'last_frame', 'first_to_middle_diff', 'middle_to_last_diff'],
                       help='Methods to use for processing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base filename for output files
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    
    # Process with each method
    for method in args.methods:
        output_file = os.path.join(args.output_dir, f"{input_basename}_{method}_answers.jsonl")
        
        print(f"\n{'='*80}")
        print(f"Processing method: {method}")
        print(f"{'='*80}")
        
        try:
            process_jsonl_with_method(args.input, output_file, args.base_data_dir, method)
        except Exception as e:
            print(f"Error processing method {method}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("All methods completed!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 