N_EXAMPLES=10
THINKING_TOKENS=1500

#python compare_reasoning.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS

python compare_reasoning.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS

#python compare_reasoning.py --model "claude-3-opus-latest" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS

#python compare_reasoning.py --model "claude-3-7-sonnet-latest" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS
