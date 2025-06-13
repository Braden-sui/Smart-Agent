# Configuration Management

This document describes how to configure the `simulated-minds` application. Configuration is primarily managed through environment variables, which allows for flexibility across different deployment environments (development, testing, production).

## 1. Environment Variables

An `.env` file in the project root is used to load environment variables for local development. See the `.env.example` file for a complete template.

### Key Variables

- **`LLM_BACKEND`**: Specifies which Language Model (LLM) backend to use.
  - **`rwkv7-gguf`** (Default): Uses the `RWKV7GGUFClient`, a highly optimized, stateful client for running RWKV models locally via `llama-cpp-python`.
  - **`transformers`**: Uses a standard Hugging Face `transformers` pipeline. Less performant and lacks advanced state management.

- **`RWKV_MODEL_PATH`**: The absolute or relative path to the GGUF-formatted RWKV model file required by the `RWKV7GGUFClient`.
  - **Example**: `C:/models/rwkv-v5-world-3b.gguf`

- **`LLM_CONTEXT_SIZE`**: The context window size (in tokens) for the LLM. This should be set according to the model's capabilities.
  - **Default**: `4096`

- **`MEM0_API_KEY`**: The API key for the Mem0 Cognitive Memory service. If provided, the system will use the `Mem0Client` for long-term memory storage. If not, it will fall back to a transient, in-memory store.

## 2. Loading Configuration

The application entry point (`main.py` or similar) is responsible for loading the environment variables at startup using a library like `python-dotenv`. Components like the `Planner` and `RWKV7GGUFClient` then read these variables to configure themselves.

## 3. Best Practices

- **Do Not Commit `.env`**: The `.env` file should never be committed to version control. It is for local development secrets and settings only.
- **Use `.env.example`**: The `.env.example` file should be kept up-to-date with all required environment variables, but with placeholder or empty values.
- **Production Environments**: In production, environment variables should be set directly in the deployment environment (e.g., Docker environment, systemd service file, PaaS configuration).
