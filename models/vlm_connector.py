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
from utils.data_utils import get_middle_frame, get_frame_differences, get_frame_differences_with_optical_flow


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
        
        # Enhance the question with specific instructions for VQA tasks
        enhanced_question = self._enhance_question_for_vqa(question)
                
        # Process input based on model type
        if "llava" in self.model_name.lower():
            prompt = f"<image>\nUser: {enhanced_question}\nAssistant:"
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            response = response.split("Assistant:")[1].strip()
            
            # Post-process to get more concise answers
            return self._post_process_answer(response, question)
            
        elif "blip" in self.model_name.lower():
            if "blip2" in self.model_name.lower() or "instructblip" in self.model_name.lower():
                inputs = self.processor(image, enhanced_question, return_tensors="pt").to(self.device, torch.float16)
                
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens
                    )
                    
                response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
                return self._post_process_answer(response, question)
            else:
                # Original BLIP
                inputs = self.processor(image, enhanced_question, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    output = self.model.generate(**inputs)
                    
                response = self.processor.decode(output[0], skip_special_tokens=True)
                return self._post_process_answer(response, question)
                
        elif "cogvlm" in self.model_name.lower():
            inputs = self.processor(image, enhanced_question, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            response = response.replace(enhanced_question, "").strip()
            return self._post_process_answer(response, question)
            
        else:
            # Generic approach for other models
            inputs = self.processor(image, enhanced_question, return_tensors="pt").to(self.device)
            
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
        Enhance the question with specific instructions for better VQA performance.
        """
        # Check if it's asking for expression classification
        if "coarse expression" in question.lower():
            return f"{question} Answer with only one word from: positive, negative, surprise."
        elif "fine-grained expression" in question.lower():
            return f"{question} Answer with only one word from: happiness, sadness, anger, fear, disgust, surprise."
        elif "action unit" in question.lower() and "present" in question.lower():
            return f"{question} List only the action unit names, separated by commas."
        elif question.lower().startswith("is the action unit"):
            return f"{question} Answer with only: yes or no."
        else:
            # For other questions, add general instruction for conciseness
            return f"{question} Give a brief, direct answer."
    
    def _post_process_answer(self, response, original_question):
        """
        Post-process the model response to extract the most relevant answer.
        """
        # Clean up the response
        response = response.strip()
        
        # For coarse expression questions
        if "coarse expression" in original_question.lower():
            # Look for key terms
            response_lower = response.lower()
            if "negative" in response_lower or "anger" in response_lower or "frown" in response_lower or "displeasure" in response_lower:
                return "negative"
            elif "positive" in response_lower or "happiness" in response_lower or "joy" in response_lower or "smile" in response_lower:
                return "positive"
            elif "surprise" in response_lower:
                return "surprise"
            else:
                # Return first meaningful word
                words = response.split()
                for word in words:
                    word_clean = word.lower().strip('.,!?"')
                    if word_clean in ['positive', 'negative', 'surprise', 'happiness', 'anger', 'fear', 'sadness', 'disgust']:
                        if word_clean in ['happiness', 'joy', 'smile']:
                            return "positive"
                        elif word_clean in ['anger', 'fear', 'sadness', 'disgust', 'frown']:
                            return "negative"
                        else:
                            return word_clean
        
        # For fine-grained expression questions
        elif "fine-grained expression" in original_question.lower():
            response_lower = response.lower()
            emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
            for emotion in emotions:
                if emotion in response_lower:
                    return emotion
        
        # For yes/no questions
        elif original_question.lower().startswith("is the"):
            response_lower = response.lower()
            if "yes" in response_lower or "shown" in response_lower:
                return "yes"
            elif "no" in response_lower or "not" in response_lower:
                return "no"
        
        # For action unit questions, try to extract clean list
        elif "action unit" in original_question.lower():
            # Remove common prefixes
            response = response.replace("The action units present are:", "")
            response = response.replace("The action unit present is:", "")
            response = response.strip(" :")
            
            # Clean up formatting
            if response.endswith("."):
                response = response[:-1]
        
        # Default: return first sentence or first few words if it's too long
        sentences = response.split('.')
        first_sentence = sentences[0].strip()
        
        # If still too long, get first reasonable chunk
        if len(first_sentence) > 50:
            words = first_sentence.split()
            if len(words) > 8:
                return ' '.join(words[:8])
        
        return first_sentence if first_sentence else response

    def answer_question_with_frame_difference(self, image, question, max_new_tokens=100, difference_type="first_to_middle"):
        """
        Answer a question about frame differences in an image sequence.
        
        Args:
            image: Path to directory of images or single image path
            question (str): Question to answer
            max_new_tokens (int): Maximum number of tokens to generate
            difference_type (str): Type of difference to compute ("first_to_middle", "middle_to_last", "both")
            
        Returns:
            str: Answer to the question based on frame differences
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Enhance the question with specific instructions for VQA tasks
        enhanced_question = self._enhance_question_for_vqa(question)
            
        # Handle directory of images for frame difference analysis
        if isinstance(image, str) and os.path.isdir(image):
            # Get frame differences
            frame_data = get_frame_differences(image)
            if frame_data is None:
                raise ValueError(f"Could not compute frame differences for directory: {image}")
            
            # Select which difference image to use
            if difference_type == "first_to_middle":
                diff_image = frame_data['first_to_middle_diff']
                context = "This image shows the difference between the first frame and middle frame of a micro-expression sequence. "
            elif difference_type == "middle_to_last":
                diff_image = frame_data['middle_to_last_diff']
                context = "This image shows the difference between the middle frame and last frame of a micro-expression sequence. "
            elif difference_type == "both":
                # For now, use first_to_middle as primary, but could be extended to handle both
                diff_image = frame_data['first_to_middle_diff']
                context = "This image shows frame differences from a micro-expression sequence. "
            else:
                raise ValueError(f"Invalid difference_type: {difference_type}")
            
            if diff_image is None:
                raise ValueError(f"Could not compute {difference_type} difference")
            
            # Add context to the question
            contextualized_question = context + enhanced_question
            
            # Use the difference image for inference
            image_to_process = diff_image
            
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
                
        # Process input based on model type
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