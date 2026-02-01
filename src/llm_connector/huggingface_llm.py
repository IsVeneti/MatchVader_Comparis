from typing import Type, TypeVar, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass, field
import torch
import json
import re

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
        
        # Initialize token tracker
        self.token_usage = TokenUsageTracker()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=hf_token
        ).to(self.device)
        # Good defaults
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            self.model.generation_config.eos_token_id = eos_id
            self.model.generation_config.pad_token_id = eos_id  # prevents pad=-100 issues

        self.model.eval()

    def _apply_template(self, user_text: str) -> str:
        """
        Use chat template if available; fall back to plain text otherwise.
        """
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_text

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate raw text output from a prompt (no assumptions about format).
        Returns only the newly generated text.
        """
        prompt_text = self._apply_template(prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        prompt_tokens = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=kwargs.get("do_sample", self.temperature > 0),
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.model.generation_config.pad_token_id,
            )

        gen_ids = output_ids[0, prompt_tokens:]
        completion_tokens = len(gen_ids)
        
        # Track token usage
        self.token_usage.add(prompt_tokens, completion_tokens)
        
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return decoded.strip()
    
    def get_token_stats(self) -> dict:
        """Return token usage statistics."""
        return self.token_usage.get_stats()
    
    def reset_token_stats(self):
        """Reset token usage counters."""
        self.token_usage.reset()

    @staticmethod
    def _first_json_blob(text: str) -> str:
        """
        Extract the first JSON object or array from text.
        - Handles ```json ... ``` fences.
        - Falls back to brace/bracket matching.
        Raises ValueError if nothing JSON-like is found.
        """
        # 1) Code-fence case
        m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S | re.I)
        if m:
            return m.group(1).strip()

        # 2) Any fenced block
        m = re.search(r"```(?:\w+)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
        if m:
            return m.group(1).strip()

        # 3) Brace/bracket matching from first { or [
        start = None
        for i, ch in enumerate(text):
            if ch in "{[":
                start = i
                break
        if start is None:
            raise ValueError("No JSON start character '{' or '[' found.")

        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        for j in range(start, len(text)):
            if text[j] == opening:
                depth += 1
            elif text[j] == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start:j+1]
                    return candidate.strip()

        raise ValueError("Unbalanced JSON braces/brackets in model output.")

    @staticmethod
    def _json_prompt(user_prompt: str, schema: Type[T]) -> str:
        """
        Compose a strong instruction to return *only* JSON that validates against the Pydantic schema.
        """
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            "You are a JSON generator.\n"
            "Return ONLY valid JSON with no explanation, no code fences, and no trailing text.\n"
            "The JSON must strictly validate against this schema (Pydantic v2 JSON Schema):\n"
            f"{schema_json}\n\n"
            f"User request:\n{user_prompt}"
        )

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """
        Build a JSON-only prompt from the schema, generate, extract the JSON, then validate.
        """
        formatted = self._json_prompt(prompt, schema)
        raw_output = self.generate(formatted)

        # Try to parse directly; if that fails, try to extract a JSON blob from the text.
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            json_text = self._first_json_blob(raw_output)
            data = json.loads(json_text)

        # Validate via Pydantic
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            # Bubble up with context including a snippet
            snippet = raw_output[:300].replace("\n", "\\n")
            raise ValueError(f"Pydantic validation failed: {e}\nRaw model output (first 300 chars): {snippet}")
