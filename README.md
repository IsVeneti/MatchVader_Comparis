# MatchVader Comparis

A Python application for entity matching using Large Language Models (LLMs). The system processes pairs of entities from two datasets, uses LLMs to determine if they match, and provides evaluation metrics against ground truth data.

# Table of Contents
- [MatchVader Comparis](#matchvader-comparis)
- [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Configuration Files](#configuration-files)
    - [Dataset Configuration (`configs/dataset_config.yaml`)](#dataset-configuration-configsdataset_configyaml)
    - [Task Configuration (`configs/task_config.yaml`)](#task-configuration-configstask_configyaml)
  - [Usage](#usage)
    - [Main Processing Script](#main-processing-script)
      - [Command-Line Arguments](#command-line-arguments)
      - [Examples](#examples)
      - [Output Files](#output-files)
    - [Evaluation Script](#evaluation-script)
      - [Command-Line Arguments](#command-line-arguments-1)
      - [Examples](#examples-1)
      - [Evaluation Metrics](#evaluation-metrics)
  - [Workflow Examples](#workflow-examples)
    - [Complete Workflow: Processing to Evaluation](#complete-workflow-processing-to-evaluation)
    - [Batch Processing Large Datasets](#batch-processing-large-datasets)
  - [License](#license)


## Overview

This application consists of two main components:

1. **Entity Matching (`main.py`)**: Processes pairs of entities using HuggingFace LLMs with structured output generation
2. **Evaluation (`evaluate.py`)**: Evaluates LLM predictions against ground truth and calculates performance metrics

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd entity-matching
```

2. **Install dependencies with uv**
```bash
uv sync
```

3. **Set up environment variables**
```bash
cp example.env .env
```
Then edit `.env` and add your HuggingFace token:
```env
HF_TOKEN=your_huggingface_token_here
```

4. **Prepare configuration files**
Ensure your configuration files are in the `configs/` directory:
- `configs/dataset_config.yaml`
- `configs/task_config.yaml`

## Configuration Files

### Dataset Configuration (`configs/dataset_config.yaml`)

Defines the file paths for different datasets. Each dataset contains:
- `d1`: First dataset CSV file
- `d2`: Second dataset CSV file  
- `pairs`: Candidate pairs CSV file
- `gt`: Ground truth CSV file (for evaluation)
- `pairs_with_ids`: Pairs file with ID mappings

**Example:**
```yaml
dataset_1:
  d1: "data/d1_data/rest1clean.csv"
  d2: "data/d1_data/rest2clean.csv"
  gt: "data/d1_data/gtclean_evaluator_comma.csv"
  pairs: "data/d1_data/d1_pairs.csv"
  pairs_with_ids: "data/d1_data/d1_pairs_with_ids.csv"

dataset_2:
  d1: "data/d2_data/rest1clean.csv"
  d2: "data/d2_data/rest2clean.csv"
  gt: "data/d2_data/gtclean_evaluator_comma.csv"
  pairs: "data/d2_data/d2_pairs.csv"
  pairs_with_ids: "data/d2_data/d2_pairs_with_ids.csv"
```

**Note**: Configuration paths are hardcoded in the code:
- Dataset config: `configs/dataset_config.yaml`
- Task config: `configs/task_config.yaml`

### Task Configuration (`configs/task_config.yaml`)

Defines the tasks for entity matching, including prompts, schemas, and LLM parameters.

**Example:**
```yaml
Pairs:
  schema: "schemas/matching_schema.py"
  entities: ["entity_1", "entity_2"]
  prompt: "prompts/entity_matching_prompt.txt"
  temperature: 0.5
  max_tokens: 256

DetailedMatching:
  schema: "schemas/detailed_matching_schema.py"
  entities: ["entity_1", "entity_2"]
  prompt: "prompts/detailed_matching_prompt.txt"
  temperature: 0.3
  max_tokens: 512
```

**Fields:**
- `schema`: Path to Pydantic schema class defining output structure
- `entities`: List of entity placeholders used in the prompt template
- `prompt`: Path to prompt template file
- `temperature`: LLM temperature parameter (0.0-1.0)
- `max_tokens`: Maximum tokens for LLM response

## Usage

### Main Processing Script

The main script (`main.py`) processes entity pairs using an LLM to determine matches.

#### Command-Line Arguments

```
usage: main.py [-h] [--hf-model HF_MODEL] --task TASK --dataset DATASET
               [--log-file LOG_FILE] [--log-console] [--save SAVE]
               [--partial-save PARTIAL_SAVE] [--start-index START_INDEX]
               [--count COUNT]

Run HuggingFaceLLM with structured prompt output for entity matching.

options:
  -h, --help            show this help message and exit
  --hf-model HF_MODEL   Hugging Face model name or path.
  --task TASK           Task name (e.g., Pairs) from the task config file.
  --dataset DATASET     Dataset name (e.g., dataset_1, dataset_2) from the
                        dataset config file.
  --log-file LOG_FILE   Path to log file.
  --log-console         Enable console logging (default if no log file).
  --save SAVE           Path to save results (optional).
  --partial-save PARTIAL_SAVE
                        Save results every X entries (enables partial saving).
  --start-index START_INDEX
                        Starting pair index for processing.
  --count COUNT         Number of pairs to process (default: all from start-
                        index).
```

#### Examples

**Basic usage:**
```bash
uv run python main.py --task Pairs --dataset dataset_1 --log-console
```

**Process specific range with results saving:**
```bash
uv run python main.py \
  --task Pairs \
  --dataset dataset_1 \
  --start-index 0 \
  --count 1000 \
  --save results/experiment_1 \
  --log-console
```

**Long job with partial saving:**
```bash
uv run python main.py \
  --task Pairs \
  --dataset dataset_2 \
  --save results/dataset2_run1 \
  --partial-save 100 \
  --log-file logs/run1.log \
  --log-console
```

#### Output Files

When using `--save`, the script creates:
- `results.csv`: All results including successes and failures
- `successful_results.csv`: Only successfully processed pairs
- `partial_results.csv`: Temporary file for partial saves (deleted upon completion)

**Result columns include:**
- `row_id`: Sequential row number
- `pair_index`: Original pair index
- Dataset-specific index columns (e.g., `rest1_index`, `rest2_index`)
- `entity1_formatted`, `entity2_formatted`: Formatted entity representations
- `entity1_raw`, `entity2_raw`: Raw entity data
- `prompt`: Generated prompt sent to LLM
- `response`: Raw LLM response
- `success`: Boolean indicating if processing succeeded
- Additional columns from schema (e.g., `match`, `confidence`, `reasoning`)

### Evaluation Script

The evaluation script (`evaluate.py`) compares LLM predictions against ground truth data.

#### Command-Line Arguments

```
usage: evaluate.py [-h] [--pairs PAIRS] --ground_truth GROUND_TRUTH
                   --response_csv RESPONSE_CSV [--join {inner,outer,left,right}]
                   [--output OUTPUT]

Evaluate candidate pairs against ground truth

options:
  -h, --help            show this help message and exit
  --pairs PAIRS, -p PAIRS
                        Candidate pairs CSV file
  --ground_truth GROUND_TRUTH, -gt GROUND_TRUTH
                        Ground truth CSV file
  --response_csv RESPONSE_CSV, -r RESPONSE_CSV
                        LLM response CSV file
  --join {inner,outer,left,right}, -j {inner,outer,left,right}
                        Join type for comparison (default: inner)
  --output OUTPUT, -o OUTPUT
                        Output CSV file for matches (optional)
```

#### Examples

**Basic evaluation:**
```bash
uv run python evaluate.py \
  -r results/experiment_1/results.csv \
  -p data/d1_data/d1_pairs.csv \
  -gt data/d1_data/gtclean_evaluator_comma.csv
```

**Evaluation with output file:**
```bash
uv run python evaluate.py \
  -r results/experiment_1/results.csv \
  -p data/d1_data/d1_pairs.csv \
  -gt data/d1_data/gtclean_evaluator_comma.csv \
  -o evaluation_results/exp1_metrics
```

**Different join types:**
```bash
# Inner join (only pairs present in both ground truth and predictions)
uv run python evaluate.py -r results.csv -p pairs.csv -gt gt.csv --join inner

# Outer join (all pairs from both ground truth and predictions)
uv run python evaluate.py -r results.csv -p pairs.csv -gt gt.csv --join outer
```

#### Evaluation Metrics

The evaluation script calculates and displays:
- **Confusion Matrix**: Visual representation of true/false positives/negatives
- **Precision**: Of predicted matches, how many are correct?
- **Recall**: Of actual matches, how many did we find?
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness of predictions

**Example output:**
```json
{
  "confusion_matrix": [[1567, 40], [51, 342]],
  "precision": 0.8947,
  "recall": 0.8502,
  "f1_score": 0.8719,
  "accuracy": 0.9123
}
```

## Workflow Examples

### Complete Workflow: Processing to Evaluation

**Step 1: Process entity pairs**
```bash
uv run python main.py \
  --task Pairs \
  --dataset dataset_1 \
  --save results/run_001 \
  --partial-save 50 \
  --log-console
```

**Step 2: Evaluate results**
```bash
uv run python evaluate.py \
  -r results/run_001/results.csv \
  -p data/d1_data/d1_pairs.csv \
  -gt data/d1_data/gtclean_evaluator_comma.csv \
  -o evaluation/run_001_metrics
```

**Step 3: Review outputs**
```bash
# View results
cat results/run_001/successful_results.csv

# View metrics
cat evaluation/run_001_metrics.json
```

### Batch Processing Large Datasets

Process in chunks to manage memory and save progress:

```bash
# Process first 1000 pairs
uv run python main.py \
  --task Pairs \
  --dataset dataset_2 \
  --start-index 0 \
  --count 1000 \
  --save results/batch_1 \
  --log-console

# Process next 1000 pairs
uv run python main.py \
  --task Pairs \
  --dataset dataset_2 \
  --start-index 1000 \
  --count 1000 \
  --save results/batch_2 \
  --log-console

# Combine results and evaluate
python -c "
import pandas as pd
batch1 = pd.read_csv('results/batch_1/results.csv')
batch2 = pd.read_csv('results/batch_2/results.csv')
combined = pd.concat([batch1, batch2])
combined.to_csv('results/combined_results.csv', index=False)
"

uv run python evaluate.py \
  -r results/combined_results.csv \
  -p data/d2_data/d2_pairs.csv \
  -gt data/d2_data/gtclean_evaluator_comma.csv \
  -o evaluation/combined_metrics
```

## License
