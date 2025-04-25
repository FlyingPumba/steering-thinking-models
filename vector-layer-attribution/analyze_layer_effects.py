# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
import utils
import gc
# Add argparse for model selection
parser = argparse.ArgumentParser(description="Analyze layer effects for different reasoning behaviors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    help="Model to analyze")
parser.add_argument("--n_examples", type=int, default=10,
                    help="Number of examples to analyze per label")
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load the model in 8-bit mode")
args, _ = parser.parse_known_args()

# give eample run command
# python analyze_layer_effects --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --n_examples 500 --load_in_8bit True

# %%
def find_label_positions(annotated_response, original_text, tokenizer, label):
    """Parse annotations and find token positions for each label"""
    label_positions = []
    pattern = f'\\["{label}"\\]([^\\[]+?)(?=\\[|$)'
    matches = re.finditer(pattern, annotated_response)
    thinking_tokens = tokenizer.encode(original_text)[1:]        
    
    for match in matches:

        text = match.group(1).strip()
        text_tokens = tokenizer.encode(text)[1:]

        if len(text_tokens) == 0:
            continue
        elif len(text_tokens) == 1:
            offset = 0
        elif len(text_tokens) > 1:
            offset = 1
            text_tokens = text_tokens[1:]
        
        for j in range(len(thinking_tokens) - len(text_tokens) + 1):
            if thinking_tokens[j:j + len(text_tokens)] == text_tokens:
                token_start = j-offset
                token_end = j + len(text_tokens)
                label_positions.append((token_start, token_end))
                continue
    
    return label_positions

def compute_kl_divergence_metric(logits):
    """Compute KL divergence between predicted distribution and detached version"""
    probs = F.log_softmax(logits, dim=-1)
    detached_probs = F.log_softmax(logits.detach(), dim=-1)
    return F.kl_div(probs, detached_probs, reduction='batchmean')

def analyze_layer_effects(model, tokenizer, text, label, feature_vectors, label_positions):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    
    patching_effects = [0 for _ in range(model.config.num_hidden_layers)]

    if len(label_positions) == 0:
        return None

    for pos in label_positions:
        layer_activations = []
        layer_gradients = []
        
        start, end = pos

        with model.trace() as tracer:
            with tracer.invoke(input_ids[:, :end]) as invoker:
                # Collect activations from each layer
                for layer_idx in range(model.config.num_hidden_layers):
                    layer_activations.append(model.model.layers[layer_idx].output[0].detach().cpu().save())
                    layer_gradients.append(model.model.layers[layer_idx].output[0].grad.detach().cpu().save())
                
                # Get logits for the endpoints
                logits = model.lm_head.output.save()
                
                # Compute cross entropy metric for each labeled section
                value = compute_kl_divergence_metric(logits[0, start])

                # Backward pass
                value.backward()
    
        layer_activations = [layer_activations[i].value for i in range(model.config.num_hidden_layers)]
        layer_gradients = [layer_gradients[i].value for i in range(model.config.num_hidden_layers)]

        feature_activation = feature_vectors[label].to(torch.bfloat16).to("cuda")

        for layer_idx in range(model.config.num_hidden_layers):
            # Get activations and gradients for the entire labeled section
            activations = layer_activations[layer_idx][0, start-1:min(start, end-2)].to("cuda")
            gradients = layer_gradients[layer_idx][0, start-1:min(start, end-2)].to("cuda")
            
            effect = torch.einsum('d,sd->s', feature_activation[layer_idx], gradients).mean().abs()
            
            patching_effects[layer_idx] += effect.cpu().item()
            
            # Clean up layer-specific tensors
            del activations
            del gradients
          
        # Clean up batch-specific tensors
        del layer_activations
        del layer_gradients
        del feature_activation
        torch.cuda.empty_cache()
        gc.collect()

    patching_effects = [effect / len(label_positions) for effect in patching_effects]

    return patching_effects

def plot_layer_effects(layer_effects, model_name):
    # Set up the figure with subplots
    n_labels = sum(1 for label, effects in layer_effects.items() if effects)
    fig, axes = plt.subplots(1, n_labels, figsize=(6*n_labels, 8), facecolor='white')
    
    # Handle the case where there's only one subplot
    if n_labels == 1:
        axes = [axes]
    
    # Color scheme
    colors = ['#2E86C1', '#E67E22', '#27AE60', '#C0392B']
    
    # Get model ID for title
    model_id = model_name.split('/')[-1]
    
    # Counter for valid labels
    valid_label_idx = 0
    
    for (label, effects), color in zip(layer_effects.items(), colors):
        if not effects:  # Skip if no effects for this label
            continue
            
        # Get current axis
        ax = axes[valid_label_idx]
        ax.set_facecolor('white')
        
        effects_array = np.array(effects)
        
        # Handle NaN values by replacing them with 0
        effects_array = np.nan_to_num(effects_array, nan=0.0)
        
        # Compute mean and std, ignoring NaN values
        mean_effects = np.nanmean(effects_array, axis=0)
        std_effects = np.nanstd(effects_array, axis=0)
        
        # Apply smoothing using convolution
        window_size = 1  # Increase coarseness by reducing window size
        kernel = np.ones(window_size) / window_size
        smoothed_effects = np.convolve(mean_effects, kernel, mode='valid')
        std_smoothed = np.convolve(std_effects, kernel, mode='valid')
        
        x = range(len(smoothed_effects))
        
        ax.fill_between(x, 
                        smoothed_effects - std_smoothed,
                        smoothed_effects + std_smoothed,
                        alpha=0.2, 
                        color=color)
        
        ax.plot(x, smoothed_effects, 
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4)
        
        # Set title and labels for each subplot
        ax.set_title("{}".format(label.replace('-', ' ').title()), 
                    fontsize=20, 
                    pad=10, 
                    color='black')
        
        ax.set_xlabel('Layer', fontsize=16, labelpad=10, color='black')
        
        # Only set y-label for the first subplot
        if valid_label_idx == 0:
            ax.set_ylabel('Mean KL-Divergence', fontsize=16, labelpad=10, color='black')
        
        ax.tick_params(axis='both', which='major', labelsize=14, colors='black')
        
        # Remove offset on x-axis
        ax.margins(x=0)
        
        # Add box and grid with stronger visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # Make the box lines thicker
            spine.set_color('black')  # Set explicit color
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        # Enhanced grid settings
        ax.grid(True, 
                linestyle='--',      # Dashed lines
                alpha=0.4,           # More opaque
                color='gray',        # Gray color
                which='major')       # Show major grid lines
        
        valid_label_idx += 1
    
    # Add a common title for all subplots
    fig.suptitle(model_id, fontsize=24, y=0.98, color='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
    
    model_id_lower = model_name.split('/')[-1].lower()
    
    plt.savefig(f'results/figures/layer_effects_{model_id_lower}_subplots.pdf', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.show()
    plt.close()

# %%
# Load model and data
model_name = args.model
print(f"Loading model {model_name}...")
model, tokenizer, feature_vectors = utils.load_model_and_vectors(load_in_8bit=args.load_in_8bit, compute_features=True, model_name=model_name)

# Create directories
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# %%
# Get model identifier for file naming
model_id = model_name.split('/')[-1].lower()
responses_path = f'../train-steering-vectors/results/vars/responses_{model_id}.json'

with open(responses_path, 'r') as f:
    results = json.load(f)

# %%
labels = [
    'backtracking',
    'uncertainty-estimation',
    'example-testing',
    'adding-knowledge',
]
n_examples = args.n_examples  # Number of examples to analyze per label

# Store results
layer_effects = {label: [] for label in labels}

# Analyze each label
for label in labels:
    print(f"Analyzing label: {label}")
    for example in tqdm(results[:n_examples]):
        original_text = example['full_response']
        annotated_text = example['annotated_thinking']

        
        # Find token positions of labeled sentences
        label_positions = []
        for label_j in labels:
            if label_j != label:
                label_positions.extend(find_label_positions(annotated_text, original_text, tokenizer, label_j))

        if label_positions:  # Only process if we found labeled sentences
            effects = analyze_layer_effects(
                model,
                tokenizer,
                original_text,
                label,
                feature_vectors,
                label_positions
            )

            if effects:
                layer_effects[label].append(effects)

# %% Plot results
plot_layer_effects(layer_effects, model_name)

# %%
