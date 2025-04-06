from llama_cpp import Llama
from pydantic import BaseModel, field_validator
from typing import List, Any, Type, TypeVar, Optional, Generic

T = TypeVar("T", bound=BaseModel)

class LlamaStringResponse(BaseModel):
    response: str

    @field_validator("response")
    @classmethod
    def validate_response(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Empty response")
        return value

class LlamaClient:
    def __init__(self, model_path: str, n_ctx: int = 4096, verbose: bool = False):
        """
        Initializes the Llama model.

        Args:
            model_path (str): Path to the .gguf Llama model file.
            n_ctx (int): Context window size for the model.
            verbose (bool): Verbosity flag for debugging.
        """
        self.model_path = model_path
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=verbose)

    def prompt(self, prompt: str, response_model: Type[T] = LlamaStringResponse,
        max_tokens: int = 4096, stop: List[str] = ["Q:", "\n"],
        temperature: float = 0.1, max_retries: int = 3) -> Optional[T]:
        """_summary_

        Args:
            prompt (str): Prompt string to send to the model.
            response_model (Type[T], optional): Pydantic model to validate the response. Defaults to LlamaStringResponse.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
            stop (_type_, optional): List of stop sequences. Defaults to ["Q:", "\n"].
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_retries (int, optional): Retry count on empty or irregular response. Defaults to 3.

        Returns:
            Optional[T]: Validated response object using pydantic or None if validation fails.
        """
    
        full_prompt = f"Q: {prompt} \nA:"
        for attempt in range(max_retries):
            try:
                result = self.llm(prompt=full_prompt, max_tokens=max_tokens, stop=stop, temperature=temperature)
                output_text = result["choices"][0]["text"].strip()

                # Parse with dynamic model
                return response_model.model_validate_json(output_text)
            
            except ValueError:
                if attempt == max_retries - 1:
                    return {"error": "Empty response after retries."}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}


    def chat(self, prompt: str, response_model: Type[T] = LlamaStringResponse,
            system_role: str = "", max_tokens: int = 4096,
            temperature: float = 0.1, max_retries: int = 3) -> Optional[T]:
        """_summary_

        Args:
            prompt (str): Prompt string to send to the model.
            response_model (Type[T], optional): Pydantic model to validate the response. Defaults to LlamaStringResponse.
            system_role (str, optional): System role for the chat model. Defaults to "".
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4096.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            max_retries (int, optional): Retry count on empty or irregular response. Defaults to 3.

        Returns:
            Optional[T]: _description_
        """
        
        messages = [{"role": "system", "content": system_role}] if system_role else []
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                result = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
                output_text = result["choices"][0]["message"]["content"].strip()

                return response_model.model_validate_json(output_text)
            
            except ValueError:
                if attempt == max_retries - 1:
                    return {"error": "Empty response after retries."}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)}"}

