#!/bin/bash

# Run MATH evaluation for DeepSeek-R1-Distill-Llama-8B model
echo "Running MATH evaluation for DeepSeek-R1-Distill-Llama-8B..."
python evaluate_MATH.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --n_examples 50

# Run MATH evaluation for DeepSeek-R1-Distill-Qwen-14B model
echo "Running MATH evaluation for DeepSeek-R1-Distill-Qwen-14B..."
python evaluate_MATH.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --n_examples 50

echo "MATH evaluations completed." 