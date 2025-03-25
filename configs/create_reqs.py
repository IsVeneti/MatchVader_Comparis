
import subprocess
import os

from config import REQUIRMENTS_PATH, ROOT_DIR, VENV_DIR

def run_pipreqs():
    """
    Runs pipreqs on the project directory while ignoring the virtual environment directory.
    """
    subprocess.run([
        "pipreqs",
        ROOT_DIR,
        "--force",
        "--ignore",
        VENV_DIR
    ], check=True)
    print("pipreqs executed successfully.")

def remove_duplicate_requirements():
    """
    Removes duplicate lines from the requirements file while preserving the original order.
    """
    # Read all lines from the requirements file.
    with open(REQUIRMENTS_PATH, "r") as file:
        lines = file.readlines()
    
    # Remove duplicates while preserving order.
    unique_lines = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    
    # Write the unique lines back to the file.
    with open(REQUIRMENTS_PATH, "w") as file:
        file.write("\n".join(unique_lines) + "\n")
    print("Duplicates removed from requirements.txt.")


if __name__ == "__main__":
    run_pipreqs()
    remove_duplicate_requirements()

