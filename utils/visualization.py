import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from data_utils import get_middle_frame


def display_image_with_answer(image_path, question, answer, ground_truth=None, figsize=(8, 8), use_middle_frame=True):
    """
    Display an image with question and answer.
    
    Args:
        image_path: Path to the image or directory containing frames
        question: Question text
        answer: Model's answer
        ground_truth: Optional ground truth answer
        figsize: Figure size
        use_middle_frame: Whether to use middle frame if image_path is a directory
    """
    try:
        # Handle directory paths by getting middle frame
        if os.path.isdir(image_path) and use_middle_frame:
            middle_frame_path = get_middle_frame(image_path)
            if middle_frame_path:
                image_path = middle_frame_path
            else:
                print(f"Warning: Could not find middle frame in {image_path}")
                return
                
        image = Image.open(image_path).convert('RGB')
        
        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.axis('off')
        
        title = f"Q: {question}\nA: {answer}"
        if ground_truth:
            match = "✓" if answer.lower().strip() == ground_truth.lower().strip() else "✗"
            title += f"\nGround Truth: {ground_truth} {match}"
            
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")


def visualize_results_grid(results, data_dir, num_samples=4, cols=2, figsize=(15, 12), use_middle_frame=True):
    """
    Visualize a grid of inference results.
    
    Args:
        results: List of result dictionaries
        data_dir: Base directory for images
        num_samples: Number of samples to display
        cols: Number of columns in the grid
        figsize: Figure size
        use_middle_frame: Whether to use middle frame if image_path is a directory
    """
    samples = results[:num_samples]
    rows = (len(samples) + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    
    for i, result in enumerate(samples):
        dataset = result['dataset']
        image_id = result['image_id']
        question = result['question']
        prediction = result['prediction']
        ground_truth = result['ground_truth']
        
        # Construct image path similar to demo.py
        base_path = os.path.join(data_dir, dataset, f"{image_id[0:3]}", f"{image_id}")
        
        try:
            # Handle directory paths
            if os.path.isdir(base_path) and use_middle_frame:
                image_path = get_middle_frame(base_path)
                if not image_path:
                    raise Exception(f"Could not find middle frame in {base_path}")
            else:
                # Try with .jpg extension
                image_path = f"{base_path}.jpg"
                if not os.path.exists(image_path):
                    raise Exception(f"Image not found at {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image)
            plt.axis('off')
            
            match = "✓" if prediction.lower().strip() == ground_truth.lower().strip() else "✗"
            plt.title(f"Q: {question}\nA: {prediction}\nGT: {ground_truth} {match}", fontsize=10)
            
        except Exception as e:
            plt.subplot(rows, cols, i + 1)
            plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_attention_heatmap(image_path, attention_weights, figsize=(12, 5), use_middle_frame=True):
    """
    Visualize attention weights as a heatmap overlaid on the image.
    
    Args:
        image_path: Path to the image or directory containing frames
        attention_weights: Attention weights tensor or array (H, W)
        figsize: Figure size
        use_middle_frame: Whether to use middle frame if image_path is a directory
    """
    # Handle directory paths by getting middle frame
    if os.path.isdir(image_path) and use_middle_frame:
        middle_frame_path = get_middle_frame(image_path)
        if middle_frame_path:
            image_path = middle_frame_path
        else:
            print(f"Warning: Could not find middle frame in {image_path}")
            return
    
    # Load and prepare image
    image = Image.open(image_path).convert('RGB')
    
    # If attention is a torch tensor, convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Resize attention to match image dimensions if needed
    if attention_weights.shape != (image.height, image.width):
        from PIL import Image as PILImage
        attention_resized = PILImage.fromarray(
            (attention_weights * 255).astype(np.uint8)
        ).resize((image.width, image.height), PILImage.BICUBIC)
        attention_weights = np.array(attention_resized) / 255.0
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Display the original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Display the attention heatmap
    img_array = np.array(image)
    
    # Create a custom colormap for the heatmap
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]  # From transparent to red with alpha
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    ax2.imshow(img_array)
    heatmap = ax2.imshow(attention_weights, alpha=0.5, cmap=cmap)
    ax2.set_title("Attention Heatmap")
    ax2.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight")
    
    plt.tight_layout()
    plt.show()


def plot_comparison_bar_chart(results, metric='exact_match', figsize=(10, 6)):
    """
    Plot comparison bar chart of model performance.
    
    Args:
        results: Dictionary mapping model names to their evaluation metrics
        metric: Metric to plot ('exact_match', 'accuracy', 'f1', etc.)
        figsize: Figure size
    """
    models = list(results.keys())
    values = [results[model]['overall'][metric] for model in models]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(models, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha='center'
        )
    
    plt.ylim(0, max(values) + 0.1)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"Model Comparison by {metric.replace('_', ' ').title()}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def save_visualization(fig, filename, dpi=300):
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution in dots per inch
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight') 