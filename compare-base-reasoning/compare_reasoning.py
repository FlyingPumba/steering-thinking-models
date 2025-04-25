# %%
import dotenv
dotenv.load_dotenv("../.env")

import argparse
import json
import random
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import torch

from utils import chat
from messages import messages

# Model Configuration
MODEL_CONFIG = {
    # API Models
    'API_MODELS': {
        'gpt-4o': 'GPT-4o',
        'claude-3-opus': 'Claude-3-Opus',
        'claude-3-7-sonnet': 'Claude-3-7-Sonnet',
        'gemini-2-0-think': 'Gemini-2-0-Think',
        'gemini-2-0-flash': 'Gemini-2-0-Flash',
        'deepseek-v3': 'DeepSeek-V3',
        'deepseek-r1': 'DeepSeek-R1'
    },
    
    # Local Models
    'LOCAL_MODELS': {
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': 'DeepSeek-Llama-8B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B': 'DeepSeek-Qwen-14B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-Qwen-32B'
    },
    
    # Thinking Models (for visualization grouping)
    'THINKING_MODELS': [
        'deepseek-llama-8b',
        'deepseek-qwen-14b',
        'deepseek-qwen-32b',
        'claude-3-7-sonnet',
        'gemini-2-0-think',
        'deepseek-r1'
    ]
}

def get_model_display_name(model_id):
    """Convert model ID to display name using configuration"""
    # Check API models first
    if model_id in MODEL_CONFIG['API_MODELS']:
        return MODEL_CONFIG['API_MODELS'][model_id]
    
    # Check local models
    for local_id, display_name in MODEL_CONFIG['LOCAL_MODELS'].items():
        if local_id in model_id:
            return display_name
    
    # Default case: format the model ID
    return model_id.title()

def is_api_model(model_name):
    """Check if the model is an API model"""
    return model_name in MODEL_CONFIG['API_MODELS']

def is_thinking_model(model_name):
    """Check if the model is a thinking model"""
    # Convert model_name to lowercase for case-insensitive comparison
    model_name = model_name.lower()
    
    return model_name in MODEL_CONFIG['THINKING_MODELS']

# Parse arguments
parser = argparse.ArgumentParser(description="Compare reasoning abilities between models")
parser.add_argument("--model", type=str, default="gemini-2-0-think", 
                    help="Model to evaluate (e.g., 'gpt-4o', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')")
parser.add_argument("--n_examples", type=int, default=10, 
                    help="Number of examples to use for evaluation")
parser.add_argument("--compute_from_json", action="store_true", 
                    help="Recompute scores from existing json instead of generating new responses")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_tokens", type=int, default=100, help="Number of max tokens")
parser.add_argument("--skip_viz", action="store_true", help="Skip visualization at the end")
args, _ = parser.parse_known_args()

# %%
def get_label_counts(thinking_process, tokenizer, labels, annotate_response=True):
    if annotate_response:
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
    else:
        annotated_response = thinking_process
    
    # Initialize token counts for each label
    label_counts = {label: 0 for label in labels}
    
    # Find all annotated sections
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    # Get tokens for the entire thinking process
    total = 0
    
    # Count tokens for each label
    for match in matches:
        label = match.group(1)
        text = match.group(2).strip()
        if label != "end-section" and label in labels:
            # Count tokens in this section
            label_counts[label] += 1
            total += 1
    
    # Convert to fractions
    label_fractions = {
        label: count / total if total > 0 else 0 
        for label, count in label_counts.items()
    }
            
    return label_fractions, annotated_response

def process_chat_response(message, model_name, model, tokenizer, labels):
    """Process a single message through chat function or model"""
    if is_api_model(model_name) and not is_thinking_model(model_name):
        # API model case (OpenAI models)
        response = chat(f"""
        Please answer the following question:
        
        Question:
        {message["content"]}
        
        Please format your response like this:
        <think>
        ...
        </think>
        [Your answer here]
        """,
        model=model_name,
        max_tokens=args.max_tokens
        )

        print(response)

    elif is_api_model(model_name) and is_thinking_model(model_name):
        response = chat(message["content"], model=model_name, max_tokens=args.max_tokens)
        print(response)

    elif is_local_model(model_name):       
        input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
                        
        with model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        ) as tracer:
            outputs = model.generator.output.save()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>") if "<think>" in response else 0
    try:
        think_end = response.index("</think>") if "</think>" in response else len(response)
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    label_fractions, annotated_response = get_label_counts(thinking_process, tokenizer, labels)
    
    return {
        "response": response,
        "thinking_process": thinking_process,
        "label_fractions": label_fractions,
        "annotated_response": annotated_response
    }

def plot_comparison(results_dict, labels):
    """Plot comparison between multiple models' results"""
    os.makedirs('results/figures', exist_ok=True)
    
    # Get model names and prepare data
    model_names = list(results_dict.keys())
    print(model_names)
    means_dict = {}
    
    # Separate models into thinking and non-thinking groups
    thinking_names = [name for name in model_names if is_thinking_model(name)]
    non_thinking_names = [name for name in model_names if not is_thinking_model(name)]
    
    # Calculate means for each model and label
    for model_name in model_names:
        means_dict[model_name] = []
        for label in labels:
            label_fracs = [ex["label_fractions"].get(label, 0) for ex in results_dict[model_name]]
            means_dict[model_name].append(np.mean(label_fracs))
    
    # Calculate average performance across all labels for each model
    model_avg_performance = {
        model_name: np.mean(means_dict[model_name]) 
        for model_name in model_names
    }
    
    # Sort models within each group by their average performance
    thinking_names = sorted(thinking_names, key=lambda x: model_avg_performance[x], reverse=True)
    non_thinking_names = sorted(non_thinking_names, key=lambda x: model_avg_performance[x], reverse=True)
    
    # Create bar plot with wider aspect ratio
    plt.style.use('seaborn-v0_8-paper')  # Use a clean scientific style
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(labels))
    
    # Enhanced black box around the plot with slightly thicker lines
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Define color palettes for thinking and non-thinking models with more distinct shades
    # Colors are ordered from darkest to brightest
    thinking_colors = [
        '#1565C0',  # Darkest blue
        '#1976D2',  # Dark blue
        '#1E88E5',  # Medium blue
        '#64B5F6'   # Lightest blue
    ]
    non_thinking_colors = [
        '#E65100',  # Darkest orange
        '#F57C00',  # Dark orange
        '#FF9800',  # Medium orange
        '#FFA726'   # Lightest orange
    ]
    
    # Calculate width based on number of models
    width = min(0.35, 0.8 / len(model_names))
    
    # Plot bars for each model, grouped by thinking/non-thinking
    bars_list = []
    
    # First plot thinking models
    for i, model_name in enumerate(thinking_names):
        # Use darker shades for higher performing models
        color_idx = i if i < len(thinking_colors) else len(thinking_colors) - 1
        bars = ax.bar(x + width * i, means_dict[model_name], width, 
                     label=model_name,
                     color=thinking_colors[color_idx], 
                     alpha=0.85, 
                     edgecolor='black', 
                     linewidth=1)
        bars_list.append(bars)
    
    # Add a small gap between groups
    gap = width * 0.5
    
    # Then plot non-thinking models
    for i, model_name in enumerate(non_thinking_names):
        # Use darker shades for higher performing models
        color_idx = i if i < len(non_thinking_colors) else len(non_thinking_colors) - 1
        bars = ax.bar(x + width * (i + len(thinking_names)) + gap, means_dict[model_name], width, 
                     label=model_name,
                     color=non_thinking_colors[color_idx], 
                     alpha=0.85, 
                     edgecolor='black', 
                     linewidth=1)
        bars_list.append(bars)
    
    # Improve grid and ticks
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)  # Put grid below bars
    
    # Set y-axis limit with more headroom and add label
    ymax = max([max(means) for means in means_dict.values()])
    ax.set_ylim(0, ymax * 1.15)  # Add 15% headroom
    ax.set_ylabel('Sentence Fraction', fontsize=16)  # Add y-axis label
    
    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    # Format x-axis labels more professionally
    ax.set_xticks(x)
    formatted_labels = [label.replace('-', ' ').title() for label in labels]
    formatted_labels = [label.replace(' ', '\n') for label in formatted_labels]
    ax.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Add vertical lines to separate thinking and non-thinking groups
    for label_idx in x:
        group_separator = label_idx + width * len(thinking_names) + gap/2
        ax.axvline(x=group_separator, color='gray', linestyle='--', alpha=0.3, zorder=0)
    
    # Enhance legend
    ax.legend(fontsize=16, frameon=True, framealpha=1, 
             edgecolor='black', bbox_to_anchor=(1, 1.02), 
             loc='upper right', ncol=2)
    
    # Adjust layout and save with high quality
    plt.tight_layout()
    plt.savefig(f'results/figures/reasoning_comparison_all_models.pdf', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# %% Parameters
n_examples = args.n_examples
random.seed(args.seed)
model_name = args.model
compute_from_json = args.compute_from_json

# Get a shorter model_id for file naming
if model_name.startswith("deepseek-ai/"):
    model_id = model_name.split('/')[-1].lower()
else:
    model_id = model_name.lower()

labels = ['initializing', 'deduction', 'adding-knowledge', 'example-testing', 
          'uncertainty-estimation', 'backtracking']

# Create directories
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# %% Load model and evaluate
model = None
tokenizer = None

if not is_api_model(model_name):
    # Load model using the utils function
    import utils
    print(f"Loading model {model_name}...")
    model, tokenizer, _ = utils.load_model_and_vectors(compute_features=False, model_name=model_name)

results = []

# %%
if compute_from_json:
    # Load existing results and recompute scores
    print(f"Loading existing results for {model_name}...")
    with open(f'results/vars/reasoning_comparison_{model_id}.json', 'r') as f:
        results = json.load(f)
else:
    # Run new evaluation
    print(f"Running evaluation for {model_name}...")
    
    # Randomly sample evaluation examples
    eval_indices = random.sample(range(len(messages)), n_examples)
    selected_messages = [messages[i] for i in eval_indices]
    
    # Process responses
    for message in tqdm(selected_messages, desc=f"Processing examples for {model_name}"):
        # Process response
        result = process_chat_response(message, model_name, model, tokenizer, labels)
        results.append(result)
    
    # Save results
    with open(f'results/vars/reasoning_comparison_{model_id}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up model to free memory
    if model is not None:
        del model
        torch.cuda.empty_cache()

# %% Generate visualization with all available models
if not args.skip_viz:
    # Load results for all models
    all_results = {}
    result_files = glob.glob('results/vars/reasoning_comparison_*.json')
    print(f"Found {len(result_files)} model results for visualization")
    
    for file_path in result_files:
        model_id = os.path.basename(file_path).replace('reasoning_comparison_', '').replace('.json', '')
        display_name = get_model_display_name(model_id)
        
        with open(file_path, 'r') as f:
            all_results[display_name] = json.load(f)
    
    # Generate visualization with all models
    if all_results:
        plot_comparison(all_results, labels)
    else:
        print("No results found for visualization")

# %%
