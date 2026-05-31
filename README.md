# Multi Task Chatbot

A modular Streamlit application for common NLP workflows powered by LangChain and Hugging Face Transformers. The app uses `google/flan-t5-base` by default and exposes a single interface for text generation, summarization, translation, sentiment analysis, and context-based question answering.

## Features

- Text generation for prompts, drafts, explanations, and plans
- Summarization with simple chunking for longer inputs
- Translation into common or custom target languages
- Sentiment classification with normalized output
- Question answering constrained to supplied context
- Streamlit session history with save and download options
- Environment-variable based runtime configuration

## Project Structure

```text
multi_task_chatbot/
|-- app.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- chains/
|   |-- generation.py
|   |-- qa.py
|   |-- sentiment.py
|   |-- summarization.py
|   `-- translation.py
|-- models/
|   `-- llm.py
|-- prompts/
|   |-- generation_prompt.py
|   |-- qa_prompt.py
|   |-- sentiment_prompt.py
|   |-- summary_prompt.py
|   `-- translation_prompt.py
|-- ui/
|   `-- streamlit_ui.py
`-- utils/
    |-- helpers.py
    `-- router.py
```

## Requirements

- Python 3.10 or newer recommended
- Enough disk space for the Hugging Face model download
- CPU works by default; CUDA can be used by setting `MODEL_DEVICE=0` when PyTorch with CUDA is installed

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Configuration

The app runs with defaults, but these environment variables can override runtime behavior:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `google/flan-t5-base` | Hugging Face model ID |
| `MODEL_DEVICE` | `-1` | `-1` for CPU, `0` for first CUDA GPU |
| `DEFAULT_TEMPERATURE` | `0.7` | Default generation temperature |
| `DEFAULT_MAX_NEW_TOKENS` | `256` | Default response token limit |
| `CHAT_HISTORY_FILE` | `chat_history.txt` | Local export path for saved history |
| `LOG_LEVEL` | `INFO` | Python logging level |

PowerShell example:

```powershell
$env:MODEL_DEVICE = "-1"
$env:DEFAULT_MAX_NEW_TOKENS = "256"
streamlit run app.py
```

## Run

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## Architecture Notes

- `app.py` is the Streamlit entrypoint.
- `config.py` centralizes model and application settings.
- `models/llm.py` loads and caches the Hugging Face Transformers pipeline, then wraps it for LangChain.
- `prompts/` contains task-specific prompt templates.
- `chains/` contains one focused function per NLP workflow.
- `utils/router.py` maps UI task selections to chain functions.
- `utils/helpers.py` handles session history, text chunking, filenames, and file exports.
- `ui/streamlit_ui.py` owns the Streamlit layout and user interactions.

## Notes

The first run downloads the configured model from Hugging Face and can take several minutes depending on network speed. Generated local history, virtual environments, Python caches, and model caches are intentionally excluded from Git.
