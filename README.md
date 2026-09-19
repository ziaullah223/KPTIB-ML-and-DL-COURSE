# Hugging Face Streamlit Chatbot

A simple local chatbot built with:

- Hugging Face Transformers
- `Qwen/Qwen2.5-0.5B-Instruct`
- Streamlit
- PyTorch

The model is public, so no Hugging Face access token is required.

## Requirements

- Python 3.10 or newer
- An internet connection for the first model download
- Approximately 2–3 GB of free disk space for packages, model files, and cache
- At least 4 GB RAM; more memory is preferable

The model can run on CPU. A supported GPU makes generation faster.

## Run in VS Code on Windows

Open the project folder in VS Code. Then open **Terminal → New Terminal** and run:

```powershell
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Install the packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the application:

```powershell
streamlit run app.py
```

Streamlit should open the application automatically. Otherwise, visit:

```text
http://localhost:8501
```

Stop the server by pressing `Ctrl+C` in the terminal.

## How it works

1. Streamlit receives a message through `st.chat_input`.
2. The message is stored in `st.session_state.messages`.
3. The tokenizer applies the model's chat template.
4. The Hugging Face Transformer generates new token IDs.
5. The tokenizer decodes those IDs into an assistant response.
6. Streamlit displays and stores the response.

## Important generation settings

- `max_new_tokens`: Maximum length of the new response in tokens.
- `do_sample=True`: Allows variation in generated answers.
- `temperature`: Controls how focused or varied generation is.
- `top_p`: Limits selection to a group of likely next tokens.
- `repetition_penalty`: Discourages repeated wording.

## Troubleshooting

### `streamlit` is not recognized

Make sure the virtual environment is active, or run:

```powershell
python -m streamlit run app.py
```

### The first response is slow

The model is downloaded on the first run. CPU generation is also slower than GPU generation.

### Out-of-memory error

Close other programs and shorten the response using the **Maximum new tokens** slider. The included 0.5B model is already selected to keep local requirements modest.

### Clear the downloaded model cache

Hugging Face stores downloaded models in the user cache. Do not remove it unless you intentionally want the model to download again.
