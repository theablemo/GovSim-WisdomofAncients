#!/bin/bash

# Define the base command
BASE_CMD="python run_simulation.py"

# Define models
MODELS=("gemma3" "gemma3:27b" "llama3.2" "llama3:8b" "llama3:70b")


echo "Starting simulations"
echo "----------------------------------------------------------"

# Loop through all combinations
for MODEL in "${MODELS[@]}"; do
    # Run with inheritance and social memory disabled
    echo "Running simulation with model $MODEL (inheritance and social memory disabled)"
    $BASE_CMD --llm-type ollama --model-name "$MODEL" --disable-inheritance --disable-social-memory
    echo "Completed run for $MODEL (disabled)"
    echo "----------------------------------------------------------"
    
    # Run with inheritance and social memory enabled
    echo "Running simulation with model $MODEL (inheritance and social memory enabled)"
    $BASE_CMD --llm-type ollama --model-name "$MODEL"
    echo "Completed run for $MODEL (enabled)"
    echo "----------------------------------------------------------"
done

echo "All simulations complete" 