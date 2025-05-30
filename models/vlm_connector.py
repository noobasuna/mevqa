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
from utils.data_utils import get_middle_frame, get_frame_differences, get_frame_differences_with_optical_flow, get_first_frame, get_last_frame


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
    
    def answer_question(self, image_dir, question, use_middle_frame=False, use_first_frame=False, use_last_frame=False):
        """
        Answer a question using a single frame from the image directory.
        
        Args:
            image_dir: Directory containing images
            question: Question to answer
            use_middle_frame: Use middle frame
            use_first_frame: Use first frame  
            use_last_frame: Use last frame
        """
        if use_middle_frame:
            image_path = get_middle_frame(image_dir)
        elif use_first_frame:
            image_path = get_first_frame(image_dir)
        elif use_last_frame:
            image_path = get_last_frame(image_dir)
        else:
            # Default to middle frame
            image_path = get_middle_frame(image_dir)
        
        if not image_path or not os.path.exists(image_path):
            return "Error: Could not find valid frame"
        
        return self._process_single_image(image_path, question)
    
    def answer_question_with_frame_difference(self, image_dir, question, difference_type="first_to_middle"):
        """
        Answer a question using frame differences.
        
        Args:
            image_dir: Directory containing images
            question: Question to answer
            difference_type: Type of difference ("first_to_middle" or "middle_to_last")
        """
        frame_data = get_frame_differences(image_dir)
        if not frame_data:
            return "Error: Could not compute frame differences"
        
        if difference_type == "first_to_middle":
            diff_image = frame_data['first_to_middle_diff']
        elif difference_type == "middle_to_last":
            diff_image = frame_data['middle_to_last_diff']
        else:
            return f"Error: Unknown difference type {difference_type}"
        
        if diff_image is None:
            return "Error: Could not compute frame difference"
        
        # Convert numpy array to PIL Image if needed
        if hasattr(diff_image, 'shape'):  # numpy array
            from PIL import Image
            import numpy as np
            if diff_image.dtype != np.uint8:
                diff_image = (diff_image * 255).astype(np.uint8)
            diff_image = Image.fromarray(diff_image)
        
        return self._process_image_object(diff_image, question)
    
    def _process_single_image(self, image_path, question, max_new_tokens=100):
        """
        Process a single image file with a question.
        
        Args:
            image_path: Path to the image file
            question: Question to answer
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Answer to the question
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Load image
        if isinstance(image_path, str):
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
            
        # Enhance the question with specific instructions for VQA tasks
        enhanced_question = self._enhance_question_for_vqa(question)
        
        return self._process_image_with_model(image, enhanced_question, max_new_tokens)
    
    def _process_image_object(self, image, question, max_new_tokens=100):
        """
        Process a PIL Image object with the question.
        
        Args:
            image: PIL Image object
            question: Question to answer
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Answer to the question
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Enhance the question with specific instructions for VQA tasks
        enhanced_question = self._enhance_question_for_vqa(question)
        
        return self._process_image_with_model(image, enhanced_question, max_new_tokens)
    
    def _process_image_with_model(self, image, question, max_new_tokens=100):
        """
        Process an image with the loaded model.
        
        Args:
            image: PIL Image object
            question: Question to answer
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Answer to the question
        """
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
            return self._post_process_answer(response, question)
            
        elif "blip" in self.model_name.lower():
            if "blip2" in self.model_name.lower() or "instructblip" in self.model_name.lower():
                inputs = self.processor(image, question, return_tensors="pt").to(self.device, torch.float16)
                
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens
                    )
                    
                response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
                return self._post_process_answer(response, question)
            else:
                # Original BLIP
                inputs = self.processor(image, question, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    output = self.model.generate(**inputs)
                    
                response = self.processor.decode(output[0], skip_special_tokens=True)
                return self._post_process_answer(response, question)
                
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
            return self._post_process_answer(response, question)
            
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
            return self._post_process_answer(response, question)
    
    def _enhance_question_for_vqa(self, question):
        """
        Enhance question with context for better VQA performance.
        
        Args:
            question: Original question
            
        Returns:
            str: Enhanced question
        """
        # Add context based on question type
        if "coarse expression" in question.lower():
            return f"Looking at this facial expression image, {question} Please choose from: positive, negative, or neutral."
        elif "fine-grained expression" in question.lower():
            return f"Analyzing the facial expression in detail, {question} Please identify the specific emotion shown."
        elif "action unit" in question.lower():
            return f"Examining the facial movements and muscle activations, {question} Please identify the specific facial action units."
        elif "is the action unit" in question.lower():
            return f"Looking carefully at the facial muscles and movements, {question} Please answer yes or no."
        else:
            return question
    
    def _post_process_answer(self, response, original_question):
        """
        Post-process the model response to clean it up.
        
        Args:
            response: Raw model response
            original_question: Original question for context
            
        Returns:
            str: Cleaned response
        """
        # Remove common artifacts
        response = response.strip()
        
        # Remove repeated question text
        if original_question.lower() in response.lower():
            response = response.replace(original_question, "").strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Answer:", "The answer is:", "Based on the image,", 
            "Looking at the image,", "In this image,", "I can see"
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Clean up formatting
        response = response.replace("\n", " ").strip()
        
        # Limit length if too long
        if len(response) > 200:
            response = response[:200].strip()
            # Try to end at a sentence boundary
            if '. ' in response:
                response = response[:response.rfind('. ') + 1]
        
        return response

    def answer_question_with_optical_flow(self, image, question, max_new_tokens=100, 
                                        flow_type="first_to_middle", visualization_type="color_wheel"):
        """
        Answer a question about optical flow in an image sequence.
        
        Args:
            image: Path to directory of images or single image path
            question (str): Question to answer
            max_new_tokens (int): Maximum number of tokens to generate
            flow_type (str): Type of flow to analyze ("first_to_middle", "middle_to_last")
            visualization_type (str): How to visualize flow ("color_wheel", "magnitude", "arrows")
            
        Returns:
            str: Answer to the question based on optical flow analysis
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Enhance the question with specific instructions for VQA tasks
        enhanced_question = self._enhance_question_for_vqa(question)
            
        # Handle directory of images for optical flow analysis
        if isinstance(image, str) and os.path.isdir(image):
            # Get frame data with optical flow
            frame_data = get_frame_differences_with_optical_flow(image)
            if frame_data is None:
                raise ValueError(f"Could not compute optical flow for directory: {image}")
            
            # Select which optical flow visualization to use
            if flow_type == "first_to_middle":
                if visualization_type == "color_wheel":
                    flow_image = frame_data['first_to_middle_flow_color']
                elif visualization_type == "magnitude":
                    flow_image = frame_data['first_to_middle_flow_magnitude']
                else:
                    flow_image = frame_data['first_to_middle_flow_color']  # fallback
                    
                context = f"This image shows optical flow (motion patterns) between the first frame and middle frame of a micro-expression sequence, visualized as {visualization_type}. Colors and patterns indicate direction and magnitude of facial movements. "
                
            elif flow_type == "middle_to_last":
                if visualization_type == "color_wheel":
                    flow_image = frame_data['middle_to_last_flow_color']
                elif visualization_type == "magnitude":
                    flow_image = frame_data['middle_to_last_flow_magnitude']
                else:
                    flow_image = frame_data['middle_to_last_flow_color']  # fallback
                    
                context = f"This image shows optical flow (motion patterns) between the middle frame and last frame of a micro-expression sequence, visualized as {visualization_type}. Colors and patterns indicate direction and magnitude of facial movements. "
            else:
                raise ValueError(f"Invalid flow_type: {flow_type}")
            
            if flow_image is None:
                raise ValueError(f"Could not compute {flow_type} optical flow visualization")
            
            # Add context about optical flow to the question
            contextualized_question = context + enhanced_question
            
            # Use the optical flow visualization for inference
            image_to_process = flow_image
            
        else:
            # Fallback to regular image processing
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    response = requests.get(image)
                    image_to_process = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    image_to_process = Image.open(image).convert("RGB")
            else:
                image_to_process = image
            
            contextualized_question = enhanced_question
                
        # Process input based on model type (same as frame difference method)
        if "llava" in self.model_name.lower():
            prompt = f"<image>\nUser: {contextualized_question}\nAssistant:"
            inputs = self.processor(prompt, image_to_process, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            response = response.split("Assistant:")[1].strip()
            return self._post_process_answer(response, question)
            
        elif "blip" in self.model_name.lower():
            if "blip2" in self.model_name.lower() or "instructblip" in self.model_name.lower():
                inputs = self.processor(image_to_process, contextualized_question, return_tensors="pt").to(self.device, torch.float16)
                
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens
                    )
                    
                response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
                return self._post_process_answer(response, question)
            else:
                # Original BLIP
                inputs = self.processor(image_to_process, contextualized_question, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    output = self.model.generate(**inputs)
                    
                response = self.processor.decode(output[0], skip_special_tokens=True)
                return self._post_process_answer(response, question)
                
        elif "cogvlm" in self.model_name.lower():
            inputs = self.processor(image_to_process, contextualized_question, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            response = response.replace(contextualized_question, "").strip()
            return self._post_process_answer(response, question)
            
        else:
            # Generic approach for other models
            inputs = self.processor(image_to_process, contextualized_question, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            return self._post_process_answer(response, question)

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