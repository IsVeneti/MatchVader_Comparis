from typing import Type
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, ValidationError
import torch
import json

from src.llm_interface.llm_base import AbstractLLM, SupportsStructuredOutput, T


class HuggingFaceLLM(AbstractLLM, SupportsStructuredOutput):
    """
    LLM wrapper using Hugging Face Transformers for causal language models (e.g., Mistral, LLaMA).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        device: str = None,
    ):
        """
        Args:
            model_name (str): Hugging Face model name or path (e.g., 'mistralai/Mistral-7B-Instruct-v0.1')
            temperature (float): Sampling temperature
            max_tokens (int): Max new tokens to generate
            device (str): Device to run on (e.g., 'cuda', 'cpu'). Default: auto-detect.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate raw text output from a prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            do_sample=self.temperature > 0,
        )
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded.split(prompt)[-1].strip()

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """
        Assume model returns a JSON string matching the schema. Prompt must elicit it.
        """
        raw_output = self.generate(prompt)

        try:
            data = json.loads(raw_output)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse structured output: {e}")
