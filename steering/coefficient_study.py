# %%
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
from utils import chat, steering_config
import re
import json
import random
from messages import validation_messages
import messages
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import gc
import os
import utils
import itertools

# %%
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)
    return model, tokenizer, feature_vectors

def get_label_counts(thinking_process, labels):
    # Get annotated version using chat function
    annotated_response = chat(f"""
    Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

    Available labels:
    0. initializing -> The model is rephrasing the given task and states initial thoughts.
    1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
    2. adding-knowledge -> The model is enriching the current approach with recalled facts.
    3. example-testing -> The model generates examples to test its current approach.
    4. uncertainty-estimation -> The model is stating its own uncertainty.
    5. backtracking -> The model decides to change its approach.

    The reasoning chain to analyze:
    {thinking_process}

    Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
    """)
    
    # Initialize token counts for each label
    label_counts = {label: 0 for label in messages.labels}
    
    # Find all annotated sections
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    # Get tokens for the entire thinking process
    total = 0
    
    # Count tokens for each label
    for match in matches:
        label = match.group(1)
        text = match.group(2).strip()
        if label != "end-section" and label in messages.labels:
            # Count tokens in this section
            label_counts[label] += 1
            total += 1
    
    # Convert to fractions
    label_fractions = {
        label: count / total if total > 0 else 0 
        for label, count in label_counts.items()
    }
            
    return label_fractions, annotated_response

def generate_and_analyze(model, tokenizer, message, feature_vectors, model_steering_config, label, labels, coefficient, steer_positive):
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")

    output_ids = utils.custom_generate_with_projection_removal(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=500,
        label=label if steer_positive is not None else "none",
        feature_vectors=feature_vectors if steer_positive is not None else None,
        steering_config=steering_config[model_name],
        steer_positive=steer_positive,
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    label_fractions, annotated_response = get_label_counts(thinking_process, labels)
    
    return {
        "response": response,
        "thinking_process": thinking_process,
        "label_fractions": label_fractions,
        "annotated_response": annotated_response
    }

def plot_coefficient_results(results, model_name):
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    
    # Use white background
    plt.style.use('seaborn-v0_8-white')
    
    # Create subplots for each label
    labels = list(results.keys())
    n_labels = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, label in enumerate(labels):
        ax = axes[idx]
        
        # Get coefficients and their results for both positive and negative steering
        pos_coefficients = sorted(list(results[label]["positive"].keys()))
        neg_coefficients = sorted(list(results[label]["negative"].keys()))
        
        pos_fractions = []
        neg_fractions = []
        
        for coeff in pos_coefficients:
            fracs = [ex["label_fractions"].get(label, 0) for ex in results[label]["positive"][coeff]]
            pos_fractions.append(np.mean(fracs))
            
        for coeff in neg_coefficients:
            fracs = [ex["label_fractions"].get(label, 0) for ex in results[label]["negative"][coeff]]
            neg_fractions.append(np.mean(fracs))
        
        # Calculate baseline (original) value
        baseline_fracs = [ex["label_fractions"].get(label, 0) for ex in results[label]["baseline"]]
        baseline_value = np.mean(baseline_fracs)
        
        # Add small offset to x-coordinates to prevent overlap
        offset = 0  # Small offset value
        pos_x = [float(x) + offset for x in pos_coefficients]
        neg_x = [float(x) - offset for x in neg_coefficients]
        
        # Plot lines with markers for both positive and negative steering
        ax.plot(pos_x, pos_fractions, marker='o', linewidth=2, markersize=8, 
                label='Positive Steering', color='#27AE60')
        ax.plot(neg_x, neg_fractions, marker='s', linewidth=2, markersize=8,
                label='Negative Steering', color='#E74C3C')
        
        # Add baseline horizontal line
        ax.axhline(y=baseline_value, color='#2E86C1', linestyle='--', linewidth=2, 
                   label='Original (Baseline)')
        
        # Add percentage labels on points
        for x, frac in zip(pos_x, pos_fractions):
            ax.text(x, frac, f'{frac*100:.1f}%', ha='center', va='bottom', fontsize=12)
        for x, frac in zip(neg_x, neg_fractions):
            ax.text(x, frac, f'{frac*100:.1f}%', ha='center', va='bottom', fontsize=12)
        
        # Add baseline value label
        ax.text(ax.get_xlim()[1], baseline_value, f'  {baseline_value*100:.1f}%', 
                va='center', fontsize=12, color='#2E86C1')
        
        # Improve styling
        ax.set_xlabel('Steering Coefficient', fontsize=16)
        ax.set_ylabel('Average Sentence Fraction (%)', fontsize=16)
        ax.set_title(label.replace('-', '\n'), fontsize=20, pad=20)
        ax.tick_params(axis='both', labelsize=14)
        
        # Convert y-axis to percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Show all spines
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        
        # Add legend
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'figures/coefficient_study_{model_id}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% Parameters
n_examples = 20
random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load model and vectors
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
args, _ = parser.parse_known_args()

model_name = args.model
model_id = model_name.split('/')[-1].lower()

# %%
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)

# %% Randomly sample evaluation examples
eval_indices = random.sample(range(len(validation_messages)), n_examples)

# Define coefficients to test for positive and negative steering
if "8b" in args.model.lower():
    pos_coefficients = [0.5, 1, 1.5, 2, 2.5]  # Positive steering coefficients
    neg_coefficients = [0.5, 1, 1.5, 2, 2.5]  # Negative steering coefficients
elif "14b" in args.model.lower():
    pos_coefficients = [0.5, 1, 1.5, 2, 2.5]  # Positive steering coefficients
    neg_coefficients = [0.5, 1, 1.5, 2, 2.5]  # Negative steering coefficients

# Store results
labels = ['backtracking', 'adding-knowledge', 'example-testing', 'uncertainty-estimation']
results = {
    label: {
        "baseline": [],  # Add baseline results
        "positive": {coeff: [] for coeff in pos_coefficients},
        "negative": {coeff: [] for coeff in neg_coefficients}
    } for label in labels
}

# Evaluate each label and coefficient
for label in labels:
    for idx in tqdm(eval_indices, desc=f"Processing examples for {label}"):
        message = validation_messages[idx]
        
        # Get baseline (original) response
        baseline_result = generate_and_analyze(
            model, tokenizer, message, feature_vectors,
            steering_config[model_name], label, labels,
            0, None  # No steering
        )
        
        # Only proceed if original version has >5% of the target label
        results[label]["baseline"].append(baseline_result)
        
        # Test positive steering coefficients
        for coeff in pos_coefficients:
            example_result = generate_and_analyze(
                model, tokenizer, message, feature_vectors,
                steering_config[model_name], label, labels,
                coeff, True  # Positive steering
            )
            results[label]["positive"][coeff].append(example_result)
            
        # Test negative steering coefficients
        for coeff in neg_coefficients:
            example_result = generate_and_analyze(
                model, tokenizer, message, feature_vectors,
                steering_config[model_name], label, labels,
                coeff, False  # Negative steering
            )
            results[label]["negative"][coeff].append(example_result)

# Save results
with open(f'data/coefficient_study_results_{model_id}.json', 'w') as f:
    json.dump(results, f, indent=2)

# %% Plot statistics
results = json.load(open(f'data/coefficient_study_results_{model_id}.json'))
plot_coefficient_results(results, model_name) 
# %%
