from llama_cpp import Llama
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Any, List
import os

class LlamaResponse(BaseModel):
    response: str

    @field_validator("response")
    @classmethod
    def validate_response(cls, value: str) -> str:
        """Ensures response is not empty or malformed."""
        if not value.strip():
            raise ValueError("Model returned an empty response.")
        return value

def get_llama_response(model_path: str, prompt: str, max_retries: int = 3) -> Any:
    """
    Generates a response using a specified Llama model while ensuring output validity with Pydantic.
    If an empty response is received, it retries up to `max_retries` times.
    
    Parameters:
        model_path (str): Path to the Llama model file.
        prompt (str): The input prompt to generate a response for.
        max_retries (int): Number of times to retry if an empty response is received.
    
    Returns:
        LlamaResponse or dict: A validated response object containing the model output,
        or an error message if validation fails.
    """
    llm = Llama(model_path=model_path,n_ctx=4096, verbose=False)
    prompt = "Q: " + prompt + " \nA:"
    print("Prompt: ", prompt)

    for attempt in range(max_retries):
        try:
            # Generate response with `echo=False` to prevent prompt repetition
            response = llm(prompt, max_tokens=4096, stop = ["Q:", "\n"], temperature=0.1)
            
            # Extract response text
            output_text = response["choices"][0]["text"].strip()
            
            # Validate and return as Pydantic model
            print("response: ",response)
            print("Output_text:", output_text)
            return LlamaResponse(response=output_text)
            # return output_text
        except ValueError:
            if attempt == max_retries - 1:
                return {"error": "Model returned an empty response after multiple attempts."}
        except Exception as e:
            return {"error": "An unexpected error occurred: " + str(e)}

def get_llama_response_chatc(model_path: str, prompt: str, system_role: str = "", max_retries: int = 3) -> Any:
    """
    Generates a response using a specified Llama model in a chat-based format.
    If an empty response is received, it retries up to `max_retries` times.

    Parameters:
        model_path (str): Path to the Llama model file.
        prompt (str): The input prompt to generate a response for.
        system_role (str): The role of the system to guide the model's behavior.
        max_retries (int): Number of times to retry if an empty response is received.

    Returns:
        LlamaResponse or dict: A validated response object containing the model output,
        or an error message if validation fails.
    """
    llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)

    messages = [{"role": "system", "content": system_role}] if system_role else []
    # prompt = "Q: " + prompt + " \nA:"
    messages.append({"role": "user", "content": prompt})

    print("Chat Messages: ", messages)

    for attempt in range(max_retries):
        try:
            # Generate response using a chat-based approach
            response = llm.create_chat_completion(messages=messages, max_tokens=4096, temperature=0.1)

            # Extract response text
            output_text = response["choices"][0]["message"]["content"].strip()

            # Validate and return as Pydantic model
            print("Response:", response)
            print("Output_text:", output_text)
            return LlamaResponse(response=output_text)
        except ValueError:
            if attempt == max_retries - 1:
                return {"error": "Model returned an empty response after multiple attempts."}
        except Exception as e:
            return {"error": "An unexpected error occurred: " + str(e)}

def process_csv_and_get_responses(model_path: str, csv_path: str, output_csv: str, save_interval: int = 5) -> None:
    """
    Reads a CSV file, generates a response for each row, and saves results to a new CSV file.
    Periodically saves to the CSV file every `save_interval` iterations.

    Parameters:
        csv_path (str): Path to the input CSV file.
        output_csv (str): Path to save the output CSV file.
        save_interval (int): Number of iterations before saving intermediate results.
    """
    df = pd.read_csv(csv_path,sep='|')
    results = []
    
    for idx, row in df.iterrows():
        combined_prompt = f"{row['Comparison']}\n\n{row['Prompt']}"
        # response = get_llama_response(model_path, combined_prompt)
        response = get_llama_response(model_path,combined_prompt)
        results.append({
            "Comparison": row['Comparison'],
            "Prompt": row['Prompt'],
            "Response": response.model_dump_json() if isinstance(response, LlamaResponse) else response
        })
        # Save to CSV at the defined interval
        if (idx + 1) % save_interval == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False)
            print(f"Intermediate results saved to {output_csv} at iteration {idx + 1}"
                  )
 
    
    
    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # Example usage:
    model_path = "models/llama3_1.gguf"  # Specify the path to your model file
    # response = get_llama_response("models/llama3_1.gguf", "What is the capital of France? Answer with 1 word")
    # print(response)
    print("Here we go")
    # response = get_llama_response(model_path,"Q: {\"entity1\": {\"name\": \"new york\", \"phone_number\": \"\", \"street\": \"\"}, \"entity2\": {\"name\": \"new york noodletown\", \"phone_number\": \"212/349-0923\", \"street\": \"\"}}\nDo these entities match? Answer yes or no \n A:")
    # response1 = get_llama_response(model_path,"{\"entity1\": {\"name\": \"new york\", \"phone_number\": \"\", \"street\": \"\"}, \"entity2\": {\"name\": \"new york noodletown\", \"phone_number\": \"212/349-0923\", \"street\": \"\"}}\nDo these entities match? Answer no or yes")
    # print("The response: ",response)
    # print("The response: ",response1)

    # print(get_llama_response(model_path,"What trees keep their leaves in winter?"))
    process_csv_and_get_responses(model_path,"prompt_attempts_data/old_prompts/prompt_attempt1.csv", "prompt_results_qa.csv",5)
    # print(os.getcwd())  # Prints the current working directory

    # print("A")