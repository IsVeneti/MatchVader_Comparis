from llama_cpp import Llama
from pydantic import BaseModel, field_validator
from typing import List, Any


class LlamaResponse(BaseModel):
    response: str

    @field_validator("response")
    @classmethod
    def validate_response(cls, value: str) -> str:
        """Ensures response is not empty or malformed."""
        if not value.strip():
            raise ValueError("Model returned an empty response.")
        return value


class LlamaClient:
    def __init__(self, model_path: str, n_ctx: int = 4096, verbose: bool = False):
        """
        Initializes the Llama model.

        Args:
            model_path (str): Path to the .gguf Llama model file.
            n_ctx (int): Context window size.
            verbose (bool): Whether to print debug output from LlamaCpp.
        """
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=verbose)

    def prompt(self, prompt: str, max_tokens: int = 4096, stop: List[str] = ["Q:", "\n"], temperature: float = 0.1, max_retries: int = 3) -> Any:
        """
        Sends a standard prompt and returns a validated response.

        Args:
            prompt (str): Prompt string to send to the model.
            stop (List[str]): Stop sequences.
            temperature (float): Sampling temperature.
            max_retries (int): Retry count on empty response.

        Returns:
            LlamaResponse or dict with error.
        """
        full_prompt = f"Q: {prompt} \nA:"
        for attempt in range(max_retries):
            try:
                result = self.llm(prompt=full_prompt, max_tokens=max_tokens, stop=stop, temperature=temperature)
                output_text = result["choices"][0]["text"].strip()
                return LlamaResponse(response=output_text)
            except ValueError:
                if attempt == max_retries - 1:
                    return {"error": "Empty response after retries."}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}

    def chat(self, prompt: str, system_role: str = "", max_tokens: int = 4096, temperature: float = 0.1, max_retries: int = 3) -> Any:
        """
        Sends a chat-style prompt to the model.

        Args:
            prompt (str): User message.
            system_role (str): Optional system message.
            temperature (float): Sampling temperature.

        Returns:
            LlamaResponse or dict with error.
        """
        messages = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                result = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
                output_text = result["choices"][0]["message"]["content"].strip()
                return LlamaResponse(response=output_text)
            except ValueError:
                if attempt == max_retries - 1:
                    return {"error": "Empty response after retries."}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}
