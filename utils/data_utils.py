import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path


def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_middle_frame(folder_path, file_extension='.jpg'):
    """
    Get the middle frame from a directory of images.
    
    Args:
        folder_path: Path to the folder containing images
        file_extension: Extension of image files to consider
        
    Returns:
        str: Path to the middle frame, or None if no images found
    """
    if not os.path.isdir(folder_path):
        # If it's already a file path, just return it
        if os.path.isfile(folder_path):
            return folder_path
        return None
        
    # Get all image files with the specified extension
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(file_extension)]
    
    if not image_files:
        return None
        
    # Sort the files by name
    image_files.sort()
    
    # Get the middle frame
    middle_index = len(image_files) // 2
    middle_frame = image_files[middle_index]
    
    return os.path.join(folder_path, middle_frame)


class MEVQADataset(Dataset):
    """Dataset for Micro-Expression Visual Question Answering."""
    
    def __init__(self, jsonl_file, data_dir, transform=None, processor=None, use_middle_frame=False):
        """
        Args:
            jsonl_file: Path to the JSONL file with annotations
            data_dir: Directory with dataset images
            transform: Optional transform to apply to images
            processor: Optional processor for VLM model
            use_middle_frame: Whether to use the middle frame from image folders
        """
        self.data = load_jsonl(jsonl_file)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.processor = processor
        self.use_middle_frame = use_middle_frame
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        dataset_name = item['dataset']  # casme2, samm, or smic
        image_id = item['image_id']
        question = item['question']
        answer = item['answer']
        
        # Construct image path based on dataset
        image_path = os.path.join(self.data_dir, dataset_name, f"{image_id}")
        
        # If using middle frame and image_path is a directory, get the middle frame
        if self.use_middle_frame:
            # If image_id doesn't end with a file extension, check if it's a directory
            if not image_id.lower().endswith(('.jpg', '.jpeg', '.png')):
                dir_path = image_path
                if os.path.isdir(dir_path):
                    middle_frame_path = get_middle_frame(dir_path)
                    if middle_frame_path:
                        image_path = middle_frame_path
                # If it's not a directory or no middle frame is found, append .jpg
                elif not os.path.exists(image_path):
                    image_path = f"{image_path}.jpg"
            
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            if self.processor:
                # For VLM processing (combines image and text)
                inputs = self.processor(images=image, text=question, return_tensors="pt", padding="max_length", truncation=True)
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
                inputs['answer'] = answer
                return inputs
            
            # Default return (for custom processing)
            return {
                'image': image,
                'question': question,
                'answer': answer,
                'image_id': image_id,
                'dataset': dataset_name
            }
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder or handle the error
            if self.processor:
                # For VLM with dummy image
                dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                inputs = self.processor(images=dummy_image, text=question, return_tensors="pt", padding="max_length", truncation=True)
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs['answer'] = answer
                return inputs
            
            return {
                'image': None,
                'question': question,
                'answer': answer,
                'image_id': image_id, 
                'dataset': dataset_name,
                'error': str(e)
            }


def get_dataloader(jsonl_file, data_dir, processor=None, batch_size=16, shuffle=True, num_workers=4, use_middle_frame=False):
    """Create a DataLoader for ME-VQA dataset."""
    dataset = MEVQADataset(jsonl_file, data_dir, processor=processor, use_middle_frame=use_middle_frame)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def analyze_dataset(jsonl_file):
    """Analyze the dataset for statistics."""
    data = load_jsonl(jsonl_file)
    
    # Extract unique values
    datasets = set(item['dataset'] for item in data)
    subjects = set(item['subject'] for item in data)
    questions = set(item['question'] for item in data)
    answers = set(item['answer'] for item in data)
    
    # Count by category
    dataset_counts = {}
    for d in datasets:
        dataset_counts[d] = len([item for item in data if item['dataset'] == d])
        
    question_counts = {}
    for q in questions:
        question_counts[q] = len([item for item in data if item['question'] == q])
        
    answer_counts = {}
    for a in answers:
        answer_counts[a] = len([item for item in data if item['answer'] == a])
        
    # Print statistics
    print(f"Total samples: {len(data)}")
    print(f"Unique datasets: {len(datasets)}")
    print(f"Unique subjects: {len(subjects)}")
    print(f"Unique questions: {len(questions)}")
    print(f"Unique answers: {len(answers)}")
    
    print("\nDataset distribution:")
    for k, v in dataset_counts.items():
        print(f"  - {k}: {v} samples ({v/len(data)*100:.1f}%)")
        
    print("\nTop questions:")
    for k, v in sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - \"{k}\": {v} samples")
        
    print("\nTop answers:")
    for k, v in sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - \"{k}\": {v} samples")
        
    return {
        "total_samples": len(data),
        "datasets": datasets,
        "dataset_counts": dataset_counts,
        "questions": questions,
        "question_counts": question_counts,
        "answers": answers,
        "answer_counts": answer_counts
    } 