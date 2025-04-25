# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
from datasets import load_dataset
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
from collections import defaultdict

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate steering effects on math problem solving")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to evaluate")
parser.add_argument("--n_examples", type=int, default=50,
                    help="Number of examples to evaluate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
args = parser.parse_args()

# %%
def extract_answer(response):
    """Extract the final answer from the model's response."""
    try:
        # Look for the answer after ####
        answer = response.split("</think>")[-1].strip()
        # Try to convert to float
        return answer
    except:
        return None

def evaluate_answer(question,model_answer, correct_answer):
    """Use chat API to evaluate if the answer is correct."""
    evaluation_prompt = f"""
    Consider the following question with the given correct answer:
    Question: {question}
    Correct answer: {correct_answer}

    Is the following written out response to the question arriving at the correct answer?
    Response: {model_answer}

    Respond with only "correct" or "incorrect".
    """
    
    response = utils.chat(evaluation_prompt)
    return response.strip().lower() == "correct"

def generate_and_evaluate(model, tokenizer, question, feature_vectors, model_steering_config, label, steer_mode="none"):
    """Generate and evaluate a single response."""
    message = {"role": "user", "content": question}
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    steer_positive = True if steer_mode == "positive" else False

    if steer_mode != "none":
        pos_layers = model_steering_config[label]["pos_layers"]
        neg_layers = model_steering_config[label]["neg_layers"]
        pos_coefficient = model_steering_config[label]["pos_coefficient"]
        neg_coefficient = model_steering_config[label]["neg_coefficient"]
    else:
        pos_layers = None
        neg_layers = None
        pos_coefficient = None
        neg_coefficient = None

    output_ids = utils.custom_generate_with_projection_removal(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=1000,
        label=label if steer_mode != "none" else "none",
        feature_vectors=feature_vectors,
        pos_layers=pos_layers,
        neg_layers=neg_layers,
        coefficient=pos_coefficient if steer_positive else neg_coefficient,
        steer_positive=steer_positive,
        show_progress=False
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    extracted_answer = extract_answer(response)

    return {
        "response": response,
        "extracted_answer": extracted_answer
    }

def calculate_thinking_length(response):
    """Calculate the length of thinking process between <think> and </think> tags."""
    start_idx = response.find("<think>")
    try:
        end_idx = response.find("</think>")
        if start_idx != -1 and end_idx != -1:
            thinking_text = response[start_idx + 7:end_idx].strip()
            return len(thinking_text.split())
    except:
        pass

    return len(response[start_idx + 7:].strip())

def plot_results(results, model_name):
    """Plot the evaluation results for all labels."""
    os.makedirs('results/figures', exist_ok=True)
    model_id = model_name.split('/')[-1].lower()
    
    # Get all labels from the results
    labels = list(results[0].keys())
    steering_modes = ['none', 'positive', 'negative']
    
    # Create subplots for accuracy and thinking length
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Calculate accuracy and thinking length for each label and steering mode
    accuracies = {mode: [] for mode in steering_modes}
    thinking_lengths = {mode: [] for mode in steering_modes}
    
    for label in labels:
        for mode in steering_modes:
            # Calculate accuracy
            correct = sum(1 for r in results if r[label][mode]["correct"])
            accuracy = correct / len(results)
            accuracies[mode].append(accuracy)
            
            # Calculate average thinking length
            lengths = [calculate_thinking_length(r[label][mode]["response"]) for r in results]
            avg_length = sum(lengths) / len(lengths)
            thinking_lengths[mode].append(avg_length)
    
    # Plot accuracy bars
    for i, mode in enumerate(steering_modes):
        offset = width * (i - 1)
        bars = ax1.bar(x + offset, accuracies[mode], width, 
                      label=mode.capitalize() if mode != "none" else "Original",
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%',
                    ha='center', va='bottom', fontsize=10)
    
    # Plot thinking length bars
    for i, mode in enumerate(steering_modes):
        offset = width * (i - 1)
        bars = ax2.bar(x + offset, thinking_lengths[mode], width,
                      label=mode.capitalize() if mode != "none" else "Original",
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add length labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10)
    
    # Improve styling for accuracy plot
    ax1.set_ylabel('Accuracy (%)', fontsize=16, labelpad=10)
    ax1.set_title(f'GSM8K Evaluation - {model_name}', fontsize=20, pad=20, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([label.replace('-', ' ').title() for label in labels], fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=12)
    
    # Improve styling for thinking length plot
    ax2.set_ylabel('Average Thinking Length (words)', fontsize=16, labelpad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([label.replace('-', ' ').title() for label in labels], fontsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'results/figures/gsm8k_results_{model_id}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %%
# Parameters
n_examples = args.n_examples
random.seed(args.seed)
model_name = args.model
model_id = model_name.split('/')[-1].lower()

# Create directories
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Load model and vectors
print(f"Loading model {model_name}...")
model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)

# %% Load GSM8K dataset
test_dataset = load_dataset("HuggingFaceH4/MATH-500", "test", streaming=True).shuffle(seed=args.seed)

# %% Randomly sample evaluation examples
results = []

# Define all labels to evaluate
labels = ["backtracking", "example-testing", "adding-knowledge", "uncertainty-estimation"]

# Evaluate each example
for idx, example in tqdm(enumerate(test_dataset), desc="Processing examples"):
    if idx > n_examples:
        break

    question = example["problem"]
    correct_answer = example["answer"]
    
    example_results = {}
    
    # Generate responses for each label and steering mode
    for label in labels:
        example_results[label] = {}
        for steer_mode in ["none", "positive", "negative"]:
            response_data = generate_and_evaluate(
                model, tokenizer, question, feature_vectors,
                utils.steering_config[model_name], label,
                steer_mode
            )
            
            # Evaluate correctness
            is_correct = evaluate_answer(question, response_data["extracted_answer"], correct_answer)
            
            example_results[label][steer_mode] = {
                "response": response_data["response"],
                "extracted_answer": response_data["extracted_answer"],
                "correct": is_correct
            }
    
    results.append(example_results)

# %% Save results
with open(f'results/vars/gsm8k_evaluation_results_{model_id}.json', 'w') as f:
    json.dump(results, f, indent=2)

plot_results(results, model_name)
# %%
