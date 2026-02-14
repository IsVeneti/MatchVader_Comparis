from typing import Type, TypeVar, Any
from unittest import result
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field
import torch
import json
import re
import outlines

T = TypeVar("T", bound=BaseModel)



@dataclass
class TokenUsageTracker:
    """Track token usage across multiple LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    
    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.call_count += 1
    
    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
    
    def get_stats(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.call_count,
            "avg_prompt_tokens": round(self.prompt_tokens / self.call_count, 2) if self.call_count else 0,
            "avg_completion_tokens": round(self.completion_tokens / self.call_count, 2) if self.call_count else 0,
            "avg_total_tokens": round(self.total_tokens / self.call_count, 2) if self.call_count else 0,
        }
class HuggingFaceLLM:
    """
    LLM wrapper using Hugging Face Transformers for causal language models (e.g., Mistral, LLaMA).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
        device: str | None = None,
        hf_token: str | None = None
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("SETTING UP")
        # Initialize token tracker
        self.token_usage = TokenUsageTracker()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=hf_token
        ).to(self.device)
        # Good defaults
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            self.model.generation_config.eos_token_id = eos_id
            self.model.generation_config.pad_token_id = eos_id  # prevents pad=-100 issues

        self.model.eval()
        # Wrap with outlines
        self.model = outlines.from_transformers(self.model, self.tokenizer)

    
    def get_token_stats(self) -> dict:
        """Return token usage statistics."""
        return self.token_usage.get_stats()
    
    def reset_token_stats(self):
        """Reset token usage counters."""
        self.token_usage.reset()

    def generate(self, prompt: str, **kwargs) -> str:
        print("GENERATE")
        return self.model(prompt, max_new_tokens=kwargs.get("max_tokens", self.max_tokens))
    
    def chat_template(self, prompt:str, system_prompt:str):
        prompt = outlines.inputs.Chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ])
        return prompt

    def generate_structured(self, prompt: str, schema: Type[T], **kwargs) -> T:
        print("GENERATE STRUCTURED")
        sys_prompt = "You are an entity matching expert. Please return the required json able to validate the schema"
        generator = outlines.Generator(
            self.model,
            schema
        )
        messages = self.chat_template(prompt,sys_prompt)
        result = generator(messages)
        # Outlines returns a JSON string — parse it into the Pydantic model
        if isinstance(result, str):
            return schema.model_validate_json(result)
        return result