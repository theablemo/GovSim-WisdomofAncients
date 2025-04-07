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
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key'
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
  - `--model-name`: LLM model name (default: gpt-3.5-turbo)
  - `--temperature`: LLM temperature (default: 0.7)

- Logging settings:
  - `--log-dir`: Directory for log files (default: logs)
  - `--quiet`: Disable verbose output

### Example

Run a simulation with inheritance enabled and custom parameters:
```bash
python run_simulation.py --num-fishermen 10 --initial-fish 200 --enable-inheritance --inheritance-rate 0.8
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