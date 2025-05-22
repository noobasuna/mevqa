#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.data_utils import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ME-VQA results")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to inference results JSON file")
    parser.add_argument("--output_file", type=str, default="evaluation_metrics.json",
                        help="Path to save evaluation results")
    parser.add_argument("--by_question", action="store_true",
                        help="Calculate metrics by question type")
    parser.add_argument("--by_dataset", action="store_true",
                        help="Calculate metrics by dataset")
    return parser.parse_args()


def normalize_answer(answer):
    """Normalize answer for comparison."""
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = ''.join(c for c in answer if c.isalnum() or c.isspace())
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer


def exact_match(pred, target):
    """Check if prediction is an exact match with target after normalization."""
    return normalize_answer(pred) == normalize_answer(target)


def calculate_metrics(results):
    """Calculate metrics for all results."""
    y_true = []
    y_pred = []
    exact_matches = 0
    
    for result in results:
        ground_truth = result["ground_truth"]
        prediction = result["prediction"]
        
        # Add to lists for metrics calculation
        y_true.append(ground_truth)
        y_pred.append(prediction)
        
        # Check exact match
        if exact_match(prediction, ground_truth):
            exact_matches += 1
    
    # Calculate metrics
    em_score = exact_matches / len(results)
    
    # For normalized comparison in classification metrics
    y_true_norm = [normalize_answer(ans) for ans in y_true]
    y_pred_norm = [normalize_answer(ans) for ans in y_pred]
    
    # Only calculate these if the answers are classification labels
    unique_answers = set(y_true_norm)
    if len(unique_answers) <= 10:  # Assume it's a classification task if few unique answers
        accuracy = accuracy_score(y_true_norm, y_pred_norm)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_norm, y_pred_norm, average='weighted', zero_division=0
        )
    else:
        # For open-ended questions, rely on exact match
        accuracy = em_score
        precision = recall = f1 = None
    
    return {
        "exact_match": em_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": len(results)
    }


def calculate_metrics_by_group(results, group_key):
    """Calculate metrics grouped by a specific key (e.g., question type or dataset)."""
    groups = {}
    
    # Group results
    for result in results:
        key = result[group_key] if group_key in result else result["question"]
        if key not in groups:
            groups[key] = []
        groups[key].append(result)
    
    # Calculate metrics for each group
    metrics_by_group = {}
    for key, group_results in groups.items():
        metrics_by_group[key] = calculate_metrics(group_results)
    
    return metrics_by_group


def main():
    args = parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results from {args.results_file}")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(results)
    print("\nOverall Metrics:")
    print(f"Exact Match: {overall_metrics['exact_match']:.4f}")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    if overall_metrics['precision'] is not None:
        print(f"Precision: {overall_metrics['precision']:.4f}")
        print(f"Recall: {overall_metrics['recall']:.4f}")
        print(f"F1 Score: {overall_metrics['f1']:.4f}")
    print(f"Total Samples: {overall_metrics['total_samples']}")
    
    # Calculate metrics by question type
    metrics_by_question = None
    if args.by_question:
        metrics_by_question = calculate_metrics_by_group(results, "question")
        print("\nMetrics by Question Type:")
        for question, metrics in metrics_by_question.items():
            print(f"\nQuestion: {question}")
            print(f"Exact Match: {metrics['exact_match']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            if metrics['precision'] is not None:
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Total Samples: {metrics['total_samples']}")
    
    # Calculate metrics by dataset
    metrics_by_dataset = None
    if args.by_dataset:
        metrics_by_dataset = calculate_metrics_by_group(results, "dataset")
        print("\nMetrics by Dataset:")
        for dataset, metrics in metrics_by_dataset.items():
            print(f"\nDataset: {dataset}")
            print(f"Exact Match: {metrics['exact_match']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            if metrics['precision'] is not None:
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Total Samples: {metrics['total_samples']}")
    
    # Save evaluation results
    evaluation_results = {
        "overall": overall_metrics
    }
    
    if metrics_by_question:
        evaluation_results["by_question"] = metrics_by_question
    
    if metrics_by_dataset:
        evaluation_results["by_dataset"] = metrics_by_dataset
    
    with open(args.output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {args.output_file}")


if __name__ == "__main__":
    main() 