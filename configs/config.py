import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent# This is your Project Root
# PROMPT_SETTINGS_PATH = os.path.join(ROOT_DIR, 'configs', 'prompt_settings.yaml')
MODELS_DIR= os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR,'data')
VENV_DIR = os.path.join(ROOT_DIR, ".venv")
REQUIRMENTS_PATH = os.path.join(ROOT_DIR, "requirements.txt")