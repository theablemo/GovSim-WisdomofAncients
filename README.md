# Fishing Community Simulation

This simulation models a community of fishermen who must make decisions about how many fish to catch each month, considering both their personal needs and the sustainability of the lake. The simulation uses LLMs to model the decision-making and discussion processes of the fishermen.

## Features

- Multiple fishermen making decisions about fish harvesting
- Personal memory system for each fisherman
- Social memory system for community norms
- Discussion phase with a mayor agent
- Memory inheritance between simulation runs
- Ancient of Wisdom agent for long-term community guidance
- Detailed logging and analysis capabilities

## Requirements

- Python 3.8+
- One of the following LLM providers:
  - Azure OpenAI API credentials
  - Google API key
  - Ollama (local installation)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables based on your chosen LLM provider:

### Azure OpenAI
```bash
export AZURE_OPENAI_ENDPOINT='your-azure-endpoint'
export AZURE_OPENAI_API_KEY='your-api-key'
export AZURE_OPENAI_API_VERSION='2024-12-01-preview'
export AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT_NAME='your-deployment-name'
export AZURE_OPENAI_CHAT_COMPLETION_MODEL_NAME='your-model-name'
```

### Google
```bash
export GOOGLE_API_KEY='your-api-key'
export GOOGLE_MODEL_NAME='gemini-pro'  # or your preferred model
```

### Ollama
```bash
# No environment variables needed, but ensure Ollama is installed and running locally
# Default model is 'gemma3'
```

## Usage

Run the simulation with default parameters (includes Ancient of Wisdom):
```bash
python run_simulation.py
```

To run the baseline version without Ancient of Wisdom and social memory:
```bash
python run_simulation.py --disable-inheritance --disable-social-memory
```

### Command Line Arguments

- Basic simulation parameters:
  - `--num-fishermen`: Number of fishermen (default: 5)
  - `--initial-fish`: Initial number of fish (default: 100)
  - `--collapse-threshold`: Collapse threshold (default: 5)
  - `--num-months`: Number of months per run (default: 12)
  - `--num-runs`: Number of runs (default: 5)

- Memory settings:
  - `--personal-memory-size`: Number of personal memories to retrieve (default: 5)
  - `--social-memory-size`: Number of social norms to retrieve (default: 2)

- Inheritance settings:
  - `--enable-inheritance`: Enable memory inheritance between runs
  - `--inheritance-rate`: Rate of memory inheritance (default: 0.7)

- Social memory settings:
  - `--disable-social-memory`: Disable social memory

- LLM settings:
  - `--llm-type`: LLM provider to use (choices: "openai", "google", "ollama", default: "openai")
  - `--model-name`: LLM model name (default depends on llm-type)
  - `--temperature`: LLM temperature (default: 0.0)
  - `--max-tokens`: Maximum tokens per response (default: 1000)
  - `--chunk-size`: Chunk size for embeddings (default: 1000)

- Logging settings:
  - `--log-dir`: Directory for log files (default: logs)
  - `--quiet`: Disable verbose output

### Example

Run a simulation with inheritance enabled and custom parameters:
```bash
python run_simulation.py --num-fishermen 10 --initial-fish 200 --enable-inheritance --inheritance-rate 0.8
```

Run with Ollama:
```bash
python run_simulation.py --llm-type ollama --model-name gemma3
```

## Output

The simulation generates:
1. A CSV file with detailed logs of each run
2. A summary of the simulation results including:
   - Total number of runs
   - Average fish remaining
   - Average total caught
   - Number of collapses
   - Average monthly catch

## Project Structure

- `config.py`: Configuration parameters
- `agent.py`: Fisherman agent implementation
- `mayor.py`: Mayor agent implementation
- `memory.py`: Memory system implementation
- `simulation.py`: Main simulation logic
- `run_simulation.py`: Command-line interface
- `ancient_of_wisdom.py`: Ancient of Wisdom agent implementation
- `llm_config.py`: LLM configuration and response handling 