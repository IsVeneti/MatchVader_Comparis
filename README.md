# MatchVader Comparis

A lightweight framework for structured prompt-based entity matching using Hugging Face LLMs and JSON output. Powered by `uv`, `pydantic`, and `transformers`.

---

## 📦 Setup

### 1. Install [`uv`](https://github.com/astral-sh/uv)

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
# Or on Windows (PowerShell):
iwr https://astral.sh/uv/install.ps1 -useb | iex
```

---

### 2. Create and Activate a Virtual Environment

```bash
uv venv .venv
.venv\Scripts\activate  # Windows
# OR
source .venv/bin/activate  # macOS/Linux
```

---

### 3. Install Dependencies

```bash
uv sync
```

---

## 🚀 Run the Main Script

Make sure you're in the project root (not `src/`), then run:

```bash
python src/main.py --task Pairs --hf-model deepseek-ai/deepseek-llm-7b-chat --config task_config.yaml
```

> 💡 Use `--log-console` or `--log-file logs/output.log` for logging.

---

## 🧐 Example Prompt

> Prompt entered during interactive run:

```
You are an entity matching model. Decide if the two entities refer to the same real-world object.

Answer only in JSON format: {"match": 1} if they refer to the same entity, or {"match": 0} if they do not.

Entity 1: "Apple Inc."
Entity 2: "Apple Incorporated"

Answer:
```

Expected model output:

```json
{"match": 1}
```

---

## ⚙️ Config File (`task_config.yaml`)

Example:

```yaml
Pairs:
  schema: "schemas.pairs_schema.PairsSchema"
  max_tokens: 200
  temperature: 0.2
```

---

## 🧪 Test Your Setup

```bash
uv pip list
uv pip freeze > requirements.lock
```

---

## 🔐 If Using Gated Models 

You must log in with your Hugging Face account to use gated models like Llama:

```bash
huggingface-cli login
```

---

## 📜 License

MIT license