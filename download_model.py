import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run HuggingFaceLLM with structured prompt output for entity matching.")
    parser.add_argument("--hf-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Hugging Face model name or path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Using Hugging Face model: {args.hf_model}")