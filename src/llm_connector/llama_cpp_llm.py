from typing import Any, Optional, Type, TypeVar
from llama_cpp import Llama
from pydantic import BaseModel
import json

from src.llm_interface.llm_base import AbstractLLM, SupportsStructuredOutput

T = TypeVar("T", bound=BaseModel)

class LlamaResponse(BaseModel):
    response: str


class LlamaChatLLM(AbstractLLM, SupportsStructuredOutput):
    def __init__(self, model_path: str, max_retries: int = 3, use_gpu: bool = True):
        self.model_path = model_path
        self.max_retries = max_retries
        n_gpu_layers = -1 if use_gpu else 0  # -1 = offload all layers to GPU
        self.llm = Llama(
            model_path=model_path, 
            n_ctx=4096, 
            n_gpu_layers=n_gpu_layers,  # This enables GPU
            verbose=False
        )
    def generate(self, prompt: str, system_role: str = "", **kwargs) -> str:
        """Generate a response using chat completion with retry logic."""
        messages = [{"role": "system", "content": system_role}] if system_role else []
        messages.append({"role": "user", "content": prompt})
        
        print("Chat Messages:", messages)
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm.create_chat_completion(
                    messages=messages, 
                    max_tokens=4096, 
                    temperature=0.1
                )
                
                output_text = response["choices"][0]["message"]["content"].strip()
                
                if output_text:
                    print("Response:", response)
                    print("Output_text:", output_text)
                    return output_text
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed after {self.max_retries} attempts: {str(e)}")
        
        raise ValueError("Model returned empty response after multiple attempts")
    
    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Generate a structured response matching the schema."""
        # Ask for JSON output
        json_prompt = f"{prompt}\n\nRespond with valid JSON matching this structure: {schema.schema_json()}"
        
        output = self.generate(json_prompt)
        
        # Try to parse the JSON
        try:
            data = json.loads(output)
            return schema(**data)
        except json.JSONDecodeError:
            # Try to find JSON in the output
            import re
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return schema(**data)
            raise ValueError(f"Could not parse JSON from response: {output}")