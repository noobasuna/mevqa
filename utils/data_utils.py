import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import cv2


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


def get_first_frame(folder_path, file_extension='.jpg'):
    """
    Get the first frame from a directory of images.
    
    Args:
        folder_path: Path to the folder containing images
        file_extension: Extension of image files to consider
        
    Returns:
        str: Path to the first frame, or None if no images found
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
    
    # Get the first frame
    first_frame = image_files[0]
    
    return os.path.join(folder_path, first_frame)


def get_last_frame(folder_path, file_extension='.jpg'):
    """
    Get the last frame from a directory of images.
    
    Args:
        folder_path: Path to the folder containing images
        file_extension: Extension of image files to consider
        
    Returns:
        str: Path to the last frame, or None if no images found
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
    
    # Get the last frame
    last_frame = image_files[-1]
    
    return os.path.join(folder_path, last_frame)


def compute_frame_difference(image1_path, image2_path):
    """
    Compute the difference between two images.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        
    Returns:
        PIL.Image: Difference image (image2 - image1)
    """
    try:
        # Load images
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        # Ensure images have the same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # Convert to numpy arrays
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        # Compute difference
        diff = arr2 - arr1
        
        # Normalize to 0-255 range for visualization
        # Add 128 to center the difference around gray
        diff_normalized = np.clip(diff + 128, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        diff_image = Image.fromarray(diff_normalized)
        
        return diff_image
        
    except Exception as e:
        print(f"Error computing difference between {image1_path} and {image2_path}: {e}")
        return None


def compute_optical_flow(image1_path, image2_path, flow_type="farneback"):
    """
    Compute optical flow between two images.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        flow_type: Type of optical flow ("farneback", "lucas_kanade")
        
    Returns:
        numpy.ndarray: Optical flow field (H, W, 2)
    """
    try:
        # Load images and convert to grayscale
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return None
            
        # Ensure images have the same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        if flow_type == "farneback":
            # Farneback optical flow
            flow = cv2.calcOpticalFlowPyrLK(img1, img2, None, None)
            # Actually, let's use Farneback method properly
            flow = cv2.calcOpticalFlowFarneback(
                img1, img2, None, 
                pyr_scale=0.5, 
                levels=3, 
                winsize=15, 
                iterations=3, 
                poly_n=5, 
                poly_sigma=1.2, 
                flags=0
            )
        else:
            # For Lucas-Kanade, we'd need feature points
            # This is a simplified version
            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
        return flow
        
    except Exception as e:
        print(f"Error computing optical flow between {image1_path} and {image2_path}: {e}")
        return None


def visualize_optical_flow(flow, visualization_type="color_wheel"):
    """
    Convert optical flow to a visual representation that LLaVA can interpret.
    
    Args:
        flow: Optical flow field (H, W, 2)
        visualization_type: Type of visualization ("color_wheel", "arrows", "magnitude")
        
    Returns:
        PIL.Image: Visualized optical flow
    """
    try:
        if flow is None:
            return None
            
        h, w = flow.shape[:2]
        
        if visualization_type == "color_wheel":
            # HSV color wheel representation
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        elif visualization_type == "arrows":
            # Arrow visualization
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            step = 16  # Spacing between arrows
            
            for y in range(0, h, step):
                for x in range(0, w, step):
                    dx, dy = flow[y, x]
                    if abs(dx) > 1 or abs(dy) > 1:  # Only show significant motion
                        # Draw arrow
                        cv2.arrowedLine(
                            rgb, 
                            (x, y), 
                            (int(x + dx * 5), int(y + dy * 5)),
                            (255, 255, 255), 
                            1,
                            tipLength=0.3
                        )
                        
        elif visualization_type == "magnitude":
            # Magnitude heatmap
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            rgb = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
        # Convert to PIL Image
        return Image.fromarray(rgb)
        
    except Exception as e:
        print(f"Error visualizing optical flow: {e}")
        return None


def compute_optical_flow_visualization(image1_path, image2_path, visualization_type="color_wheel"):
    """
    Compute optical flow and return a visual representation.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        visualization_type: Type of visualization
        
    Returns:
        PIL.Image: Visualized optical flow
    """
    flow = compute_optical_flow(image1_path, image2_path)
    if flow is None:
        return None
    return visualize_optical_flow(flow, visualization_type)


def get_frame_differences(folder_path, file_extension='.jpg'):
    """
    Get frame differences for a video sequence.
    
    Args:
        folder_path: Path to the folder containing images
        file_extension: Extension of image files to consider
        
    Returns:
        dict: Dictionary containing difference images and frame paths
    """
    if not os.path.isdir(folder_path):
        return None
    
    # Get frame paths
    first_frame_path = get_first_frame(folder_path, file_extension)
    middle_frame_path = get_middle_frame(folder_path, file_extension)
    last_frame_path = get_last_frame(folder_path, file_extension)
    
    if not all([first_frame_path, middle_frame_path, last_frame_path]):
        return None
    
    # Compute differences
    first_to_middle_diff = compute_frame_difference(first_frame_path, middle_frame_path)
    middle_to_last_diff = compute_frame_difference(middle_frame_path, last_frame_path)
    
    return {
        'first_frame_path': first_frame_path,
        'middle_frame_path': middle_frame_path,
        'last_frame_path': last_frame_path,
        'first_to_middle_diff': first_to_middle_diff,
        'middle_to_last_diff': middle_to_last_diff
    }


def get_frame_differences_with_optical_flow(folder_path, file_extension='.jpg'):
    """
    Enhanced version that includes both frame differences and optical flow visualizations.
    
    Args:
        folder_path: Path to the folder containing images
        file_extension: Extension of image files to consider
        
    Returns:
        dict: Dictionary containing difference images, optical flow, and frame paths
    """
    if not os.path.isdir(folder_path):
        return None
    
    # Get frame paths
    first_frame_path = get_first_frame(folder_path, file_extension)
    middle_frame_path = get_middle_frame(folder_path, file_extension)
    last_frame_path = get_last_frame(folder_path, file_extension)
    
    if not all([first_frame_path, middle_frame_path, last_frame_path]):
        return None
    
    # Compute frame differences (existing functionality)
    first_to_middle_diff = compute_frame_difference(first_frame_path, middle_frame_path)
    middle_to_last_diff = compute_frame_difference(middle_frame_path, last_frame_path)
    
    # Compute optical flow visualizations
    first_to_middle_flow_color = compute_optical_flow_visualization(
        first_frame_path, middle_frame_path, "color_wheel"
    )
    middle_to_last_flow_color = compute_optical_flow_visualization(
        middle_frame_path, last_frame_path, "color_wheel"
    )
    
    first_to_middle_flow_magnitude = compute_optical_flow_visualization(
        first_frame_path, middle_frame_path, "magnitude"
    )
    middle_to_last_flow_magnitude = compute_optical_flow_visualization(
        middle_frame_path, last_frame_path, "magnitude"
    )
    
    return {
        'first_frame_path': first_frame_path,
        'middle_frame_path': middle_frame_path,
        'last_frame_path': last_frame_path,
        'first_to_middle_diff': first_to_middle_diff,
        'middle_to_last_diff': middle_to_last_diff,
        'first_to_middle_flow_color': first_to_middle_flow_color,
        'middle_to_last_flow_color': middle_to_last_flow_color,
        'first_to_middle_flow_magnitude': first_to_middle_flow_magnitude,
        'middle_to_last_flow_magnitude': middle_to_last_flow_magnitude
    }


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