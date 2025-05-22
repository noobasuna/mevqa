# ME-VQA: Micro-Expression Visual Question Answering

This repository contains code for performing Visual Question Answering on micro-expression datasets (CASME2, SAMM, SMIC).

## Dataset
The datasets include micro-expression images with corresponding questions and answers about:
- Coarse expression classes (positive, negative, surprise)
- Fine-grained expression classes (happiness, disgust, surprise, etc.)
- Action units present in the facial expressions

## Setup

```bash
# Clone the repository
git clone https://github.com/noobasuna/ME-VQA.git
cd ME-VQA

# Install dependencies
pip install -r requirements.txt

# Download dataset images (not included in this repo)
# Place them in the structure described in the Directory Structure section
```

## Free Source Vision-Language Models

This repository includes a connector for various free-source VLMs via the HuggingFace Transformers library. Supported models include:

- **LLaVA models**: 
  - LLaVA-1.5-7B and LLaVA-1.5-13B (`llava-1.5-7b`, `llava-1.5-13b`)
  - Original LLaVA models (`llava-7b`, `llava-13b`)

- **BLIP models**: 
  - BLIP-VQA (`blip-vqa-base`)
  - BLIP-2 (`blip2-opt-2.7b`)
  - InstructBLIP (`instructblip-vicuna-7b`)

- **Other VLMs**:
  - CogVLM (`cogvlm-chat`)
  - MiniGPT-4 (experimental)

## Working with Image Directories

ME-VQA supports working with individual image files or directories containing multiple frames. When using a directory, the system can automatically select the middle frame by:

1. Sorting all image files in the directory alphabetically
2. Selecting the image at the middle index

This is useful for micro-expression datasets where each expression is captured as a sequence of frames.

To use this feature, add the `--use_middle_frame` flag when running the scripts.

## Usage

### Demo

Run a simple demo with a sample from the dataset:

```bash
python demo.py --model_name llava-1.5-7b --jsonl_file me_vqa_samm_casme2_smic.jsonl --data_dir data
```

Use a specific image folder and get the middle frame:

```bash
python demo.py --model_name llava-1.5-7b --image_folder data/casme2/sub01_EP02_01 --question "What is the coarse expression class?" --use_middle_frame
```

List available models:

```bash
python demo.py --list_models
```

Analyze the dataset:

```bash
python demo.py --analyze --jsonl_file me_vqa_samm_casme2_smic.jsonl
```

### Inference

Run inference on a single image:

```bash
python inference.py --model_name llava-1.5-7b --image_path data/casme2/sub01_EP02_01f.jpg --question "What is the coarse expression class?"
```

Run inference on an image directory (using middle frame):

```bash
python inference.py --model_name llava-1.5-7b --image_path data/casme2/sub01_EP02_01 --question "What is the coarse expression class?" --use_middle_frame
```

Run batch inference on multiple samples from a JSONL file:

```bash
python inference.py --model_name llava-1.5-7b --jsonl_file me_vqa_samm_casme2_smic.jsonl --data_dir data --max_samples 10
```

Use middle frames for batch inference:

```bash
python inference.py --model_name llava-1.5-7b --jsonl_file me_vqa_samm_casme2_smic.jsonl --data_dir data --max_samples 10 --use_middle_frame
```

### Evaluation

Evaluate the results of inference:

```bash
python evaluate.py --results_file results.json --by_question --by_dataset
```

## VLM Connector API

You can use the VLM connector in your own code:

```python
from models.vlm_connector import VLMConnector
from utils.data_utils import get_middle_frame

# Initialize a VLM
connector = VLMConnector("llava-1.5-7b")

# Answer a question about a single image
answer = connector.answer_question(
    "data/casme2/sub01_EP02_01f.jpg",
    "What is the coarse expression class?"
)

# Or use an image directory and get the middle frame
answer = connector.answer_question(
    "data/casme2/sub01_EP02_01",  # Directory with multiple frames
    "What is the coarse expression class?",
    use_middle_frame=True  # Will automatically select the middle frame
)

print(f"Answer: {answer}")
```

## Directory Structure

You can organize your data in two ways:

### Option 1: Individual Image Files

```
ME-VQA/
├── data/
│   ├── casme2/   # CASME2 dataset images
│   │   ├── sub01_EP02_01f.jpg
│   │   ├── sub01_EP02_02f.jpg
│   │   └── ...
│   ├── samm/     # SAMM dataset images
│   └── smic/     # SMIC dataset images
```

### Option 2: Image Folders (for sequences)

```
ME-VQA/
├── data/
│   ├── casme2/   # CASME2 dataset image folders
│   │   ├── sub01_EP02_01/  # Folder containing sequence frames
│   │   │   ├── frame_001.jpg
│   │   │   ├── frame_002.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── samm/
│   └── smic/
```

### Code Structure

```
ME-VQA/
├── models/
│   └── vlm_connector.py  # Connector for Vision-Language Models
├── utils/
│   ├── data_utils.py     # Data loading utilities
│   └── visualization.py  # Visualization tools
├── demo.py               # Demo script
├── inference.py          # Inference script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Dependencies
└── me_vqa_*.jsonl        # Dataset annotations
```

## Citation

If you use this code, please cite the original ME-VQA dataset papers.

## License

[MIT License] 

