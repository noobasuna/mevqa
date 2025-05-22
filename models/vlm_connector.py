import os
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    BlipProcessor, 
    BlipForQuestionAnswering,
    AutoTokenizer
)
from PIL import Image
import requests
from io import BytesIO
from utils.data_utils import get_middle_frame


class VLMConnector:
    """Connector for various Vision-Language Models."""
    
    def __init__(self, model_name=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize VLM connector.
        
        Args:
            model_name (str): Name or path of the model to use
            device (str): Device to use for inference
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        if model_name:
            self.load_model(model_name)
    
    def load_model(self, model_name):
        """
        Load a specific model.
        
        Args:
            model_name (str): Name or path of the model
        """
        self.model_name = model_name
        
        # LLaVA models
        if "llava" in model_name.lower():
            if "llava-1.5" in model_name.lower():
                # LLaVA 1.5 models
                model_id = "llava-hf/llava-1.5-7b-hf"
                if "13b" in model_name.lower():
                    model_id = "llava-hf/llava-1.5-13b-hf"
                
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).to(self.device)
                
            else:
                # Original LLaVA models  
                model_id = "liuhaotian/llava-v1-0-7b-hf"
                if "13b" in model_name.lower():
                    model_id = "liuhaotian/llava-v1-0-13b-hf"
                
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).to(self.device)
        
        # BLIP models
        elif "blip" in model_name.lower():
            model_id = "Salesforce/blip-vqa-base"
            if "blip2" in model_name.lower():
                model_id = "Salesforce/blip2-opt-2.7b-vqa"
            
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForQuestionAnswering.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)
            
        # InstructBLIP
        elif "instructblip" in model_name.lower():
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            
            model_id = "Salesforce/instructblip-vicuna-7b"
            
            self.processor = InstructBlipProcessor.from_pretrained(model_id)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)
            
        # MiniGPT-4
        elif "minigpt4" in model_name.lower():
            model_id = "Vision-CAIR/MiniGPT-4"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            ).to(self.device)
            
        # CogVLM
        elif "cogvlm" in model_name.lower():
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_id = "THUDM/cogvlm-chat-hf"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

        # Add more models as needed
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        print(f"Loaded model: {model_name} on {self.device}")
    
    def answer_question(self, image, question, max_new_tokens=100, use_middle_frame=True):
        """
        Answer a question about an image.
        
        Args:
            image: PIL image, path to image, or path to directory of images
            question (str): Question to answer
            max_new_tokens (int): Maximum number of tokens to generate
            use_middle_frame (bool): Whether to use the middle frame when given a directory
            
        Returns:
            str: Answer to the question
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Load image if path is given
        if isinstance(image, str):
            # Check if it's a directory and use_middle_frame is True
            if os.path.isdir(image) and use_middle_frame:
                middle_frame_path = get_middle_frame(image)
                if middle_frame_path:
                    image = middle_frame_path
                else:
                    raise ValueError(f"No images found in directory: {image}")
            
            # Now load the image from path
            if image.startswith(('http://', 'https://')):
                response = requests.get(image)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
                
        # Process input based on model type
        if "llava" in self.model_name.lower():
            prompt = f"<image>\nUser: {question}\nAssistant:"
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            response = response.split("Assistant:")[1].strip()
            return response
            
        elif "blip" in self.model_name.lower():
            if "blip2" in self.model_name.lower() or "instructblip" in self.model_name.lower():
                inputs = self.processor(image, question, return_tensors="pt").to(self.device, torch.float16)
                
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens
                    )
                    
                response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
                return response
            else:
                # Original BLIP
                inputs = self.processor(image, question, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    output = self.model.generate(**inputs)
                    
                response = self.processor.decode(output[0], skip_special_tokens=True)
                return response
                
        elif "cogvlm" in self.model_name.lower():
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            response = response.replace(question, "").strip()
            return response
            
        else:
            # Generic approach for other models
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            return response
            
    def available_models(self):
        """Return a list of supported models."""
        return [
            "llava-1.5-7b",
            "llava-1.5-13b",
            "blip-vqa-base",
            "blip2-opt-2.7b",
            "instructblip-vicuna-7b",
            "cogvlm-chat"
        ]


# Example usage function for testing
def test_vlm(model_name, image_path, question):
    """Test a VLM model with a given image and question."""
    connector = VLMConnector(model_name)
    answer = connector.answer_question(image_path, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    return answer 