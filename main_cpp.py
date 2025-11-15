from pydantic import BaseModel
from src.llm_connector.llama_cpp_llm import LlamaChatLLM
from llama_cpp import Llama


# llm = LlamaChatLLM(model_path="models/llama3_1.gguf",use_gpu=True)

# # Regular generation
# text = llm.generate("What is Python?")

# # Structured generation
# class Answer(BaseModel):
#     summary: str
#     confidence: float

# result = llm.generate_with_schema("Explain Python", Answe)
# print(result.summary)

llm = Llama(
    model_path="models/llama3_1.gguf",
    n_gpu_layers=-1,  # Use GPU
    verbose=True  # This will show if CUDA is being used
)

# You should see CUDA mentioned in the verbose output
response = llm("Test prompt")