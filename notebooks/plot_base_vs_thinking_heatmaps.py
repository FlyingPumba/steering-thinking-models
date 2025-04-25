# %%
import matplotlib.pyplot as plt
import torch
import os
import json
import random
import pickle
import numpy as np
from typing import List, Dict, Any, Union, Tuple


# %% Set model names
deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
original_model_name = "Qwen/Qwen2.5-14B"
# original_model_name = "Qwen/Qwen2.5-14B-Instruct"

# %%

seed = 42
random.seed(seed)

# %% Load data

annotated_responses_json_path = f"../data/annotated_responses_{deepseek_model_name.split('/')[-1].lower()}.json"
original_messages_json_path = f"../data/base_responses_{deepseek_model_name.split('/')[-1].lower()}.json"

tasks_json_path = "../data/tasks.json"

if not os.path.exists(annotated_responses_json_path):
    raise FileNotFoundError(f"Annotated responses file not found at {annotated_responses_json_path}")
if not os.path.exists(original_messages_json_path):
    raise FileNotFoundError(f"Original messages file not found at {original_messages_json_path}")
if not os.path.exists(tasks_json_path):
    raise FileNotFoundError(f"Tasks file not found at {tasks_json_path}")

print(f"Loading existing annotated responses from {annotated_responses_json_path}")
with open(annotated_responses_json_path, 'r') as f:
    annotated_responses_data = json.load(f)["responses"]
random.shuffle(annotated_responses_data)

print(f"Loading existing original messages from {original_messages_json_path}")
with open(original_messages_json_path, 'r') as f:
    original_messages_data = json.load(f)["responses"]
random.shuffle(original_messages_data)

print(f"Loading existing tasks from {tasks_json_path}")
with open(tasks_json_path, 'r') as f:
    tasks_data = json.load(f)

# %%

# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self):
        """
        Calculates the running mean, std, and sum of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = None
        self.var = None
        self.count = 0
        self.sum = None

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd()
        if self.mean is not None:
            new_object.mean = self.mean.clone()
            new_object.var = self.var.clone()
            new_object.count = float(self.count)
            new_object.sum = self.sum.clone()  # Copy sum
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        """
        self.update_from_moments(other.mean, other.var, other.count)
        if self.sum is None:
            self.sum = other.sum.clone()
        else:
            self.sum += other.sum

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = arr.double().mean(dim=0)
        batch_var = arr.double().var(dim=0)
        batch_count = arr.shape[0]
        batch_sum = arr.double().sum(dim=0)  # Calculate batch sum
        
        if batch_count == 0:
            return
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            self.sum = batch_sum  # Initialize sum
        else:
            self.update_from_moments(batch_mean, batch_var, batch_count)
            self.sum += batch_sum  # Update sum

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def compute(
        self, return_dict=False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor], Dict[str, float]]:
        """
        Compute the running mean, variance, count and sum

        Returns:
            mean, var, count, sum if return_dict=False
            dict with keys 'mean', 'var', 'count', 'sum' if return_dict=True
        """
        if return_dict:
            return {
                "mean": self.mean.item(),
                "var": self.var.item(),
                "count": self.count,
                "sum": self.sum.item(),
            }
        return self.mean, self.var, self.count, self.sum


# %% Load or collect KL stats

# Get model ID for filenames
deepseek_model_id = deepseek_model_name.split('/')[-1].lower()
original_model_id = original_model_name.split('/')[-1].lower()

# Define stats types and their corresponding dictionaries
stats_types = {
    "token": None,
    "token_pair": None,
    "next_token": None,
    "label": None,
    "next_token_and_label": None
}

# Check if all stats files exist
all_files_exist = True
for stats_type in stats_types:
    file_path = f"../data/kl_stats/kl_stats_per_{stats_type}_{deepseek_model_id}_{original_model_id}.pkl"
    if not os.path.exists(file_path):
        all_files_exist = False
        break

if all_files_exist:
    print("Loading existing KL stats from files...")
    # Load each stats file
    for stats_type in stats_types:
        file_path = f"../data/kl_stats/kl_stats_per_{stats_type}_{deepseek_model_id}_{original_model_id}.pkl"
        with open(file_path, 'rb') as f:
            stats_types[stats_type] = pickle.load(f)
        print(f"Loaded {stats_type} KL stats from {file_path}")
else:
    raise ValueError("Some stats files missing, please run the base_vs_thinking_heatmaps.py notebook first")

# %% Add visualization for token pairs and next tokens

def plot_top_stats(stats_dict, title, n=20, pair_keys=False, metric='mean', top_count_pct=0.1):
    """
    Plot statistics for tokens/pairs
    
    Args:
        stats_dict: Dictionary of statistics
        title: Title for the plot
        n: Number of top items to show
        pair_keys: Whether the keys are pairs/tuples
        metric: 'mean' or 'sum' to determine which metric to sort by
        top_count_pct: Filter to keep only top percentage by count (0.1 = top 10%)
    """
    # Create a list of (key, value, count) tuples
    values = []
    for key, stats in stats_dict.items():
        mean, _, count, sum_val = stats.compute()
        value = sum_val.item() if metric == 'sum' else mean.item()
        values.append((key, value, count))
    
    # First filter by count - keep only top percentage
    values.sort(key=lambda x: x[2], reverse=True)
    cutoff_idx = max(1, int(len(values) * top_count_pct))
    values = values[:cutoff_idx]
    
    # Then sort by the chosen metric
    values.sort(key=lambda x: x[1], reverse=True)
    top_values = values[:n]
    
    # Create lists for plotting
    if pair_keys:
        if len(top_values[0][0]) == 2:
            keys = [f"{t[0][0]}\n{t[0][1]}" for t in top_values]
        else:
            keys = [f"{t[0][0]}\n{t[0][1]}\n{t[0][2]}" for t in top_values]
    else:
        keys = [t[0] for t in top_values]
    metric_values = [t[1] for t in top_values]
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(keys)), metric_values)
    plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
    plt.title(f'{title} by {metric.capitalize()} KL Divergence (Top {n}, from top {int(top_count_pct*100)}% by count)')
    plt.xlabel('Token' if not pair_keys else 'Token Pair')
    plt.ylabel(f'{metric.capitalize()} KL Divergence')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../plots/{title.replace(' ', '_').lower()}_{metric}_{deepseek_model_id}_{original_model_id}.png")
    
    # Print the list
    print(f"\nTop {n} {title} by {metric} (filtered to top {int(top_count_pct*100)}% by count):")
    for key, value, count in top_values:
        if pair_keys:
            if len(key) == 2:
                print(f"({key[0]}, {key[1]}): {value:.4f} (count: {count})")
            else:
                print(f"({key[0]}, {key[1]}, {key[2]}): {value:.4f} (count: {count})")
        else:
            print(f"{key}: {value:.4f} (count: {count})")

# Plot all statistics with both mean and sum
plot_top_stats(stats_types["token"], "Tokens", metric='mean')
plot_top_stats(stats_types["token"], "Tokens", metric='sum')

plot_top_stats(stats_types["token_pair"], "Token Pairs", pair_keys=True, metric='mean')
plot_top_stats(stats_types["token_pair"], "Token Pairs", pair_keys=True, metric='sum')

plot_top_stats(stats_types["next_token"], "Next Tokens (Previous Token's KL)", metric='mean')
plot_top_stats(stats_types["next_token"], "Next Tokens (Previous Token's KL)", metric='sum')

plot_top_stats(stats_types["next_token_and_label"], "Next Tokens and Labels", pair_keys=True, metric='mean')
plot_top_stats(stats_types["next_token_and_label"], "Next Tokens and Labels", pair_keys=True, metric='sum')

# %% Add visualization for label statistics

def plot_label_stats(stats_dict, metric='mean', top_count_pct=0.1):
    """
    Plot statistics for labels
    
    Args:
        stats_dict: Dictionary of statistics
        metric: 'mean' or 'sum' to determine which metric to sort by
        top_count_pct: Filter to keep only top percentage by count (0.1 = top 10%)
    """
    values = []
    for label, stats in stats_dict.items():
        mean, var, count, sum_val = stats.compute()
        value = sum_val.item() if metric == 'sum' else mean.item()
        std_dev = torch.sqrt(var).item()
        values.append((label, value, count, std_dev))
    
    # First filter by count - keep only top percentage
    values.sort(key=lambda x: x[2], reverse=True)
    cutoff_idx = max(1, int(len(values) * top_count_pct))
    values = values[:cutoff_idx]
    
    # Then sort by the chosen metric
    values.sort(key=lambda x: x[1], reverse=True)
    
    # Create lists for plotting
    labels = [t[0] for t in values]
    metric_values = [t[1] for t in values]
    counts = [t[2] for t in values]
    std_devs = [t[3] for t in values]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot bars with error bars
    bars = plt.bar(range(len(labels)), metric_values, yerr=std_devs, capsize=5)
    
    # Add count annotations on top of each bar
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'n={int(count)}',
                ha='center', va='bottom')
    
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.title(f'KL Divergence by Label ({metric.capitalize()}, from top {int(top_count_pct*100)}% by count)')
    plt.xlabel('Label')
    plt.ylabel(f'{metric.capitalize()} KL Divergence')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../plots/kl_divergence_by_label_{metric}_{deepseek_model_id}_{original_model_id}.png")
    
    # Print detailed statistics
    print(f"\nLabel Statistics ({metric}, filtered to top {int(top_count_pct*100)}% by count):")
    for label, value, count, std_dev in values:
        print(f"{label:20} {value:.4f} ± {std_dev:.4f} (count: {count})")

# Plot label statistics with both metrics
plot_label_stats(stats_types["label"], metric='mean')
plot_label_stats(stats_types["label"], metric='sum')

# %% Add stacked bar plot for token pairs by label

def plot_stacked_token_pairs_by_label(
    stats_dict, 
    n=20, 
    metric='sum', 
    top_count_pct=0.1,
    ignore_categories=["initializing", "deduction"]  # New parameter
):
    """
    Create a stacked bar plot showing token pairs with different colors for each label
    
    Args:
        stats_dict: Dictionary of statistics
        n: Number of top pairs to show
        metric: Either 'sum' or 'mean' to determine which metric to use for plotting
        top_count_pct: Filter to keep only top percentage by count (0.1 = top 10%)
        ignore_categories: List of categories to ignore when calculating statistics
    """
    # First, organize data by token pairs
    pair_data = {}
    pair_counts = {}  # Track total counts for each pair
    for (current_token, next_token, label), stats in stats_dict.items():
        # Skip if label is in ignore_categories
        if label in ignore_categories:
            continue
            
        pair_key = (current_token, next_token)
        if pair_key not in pair_data:
            pair_data[pair_key] = {}
            pair_counts[pair_key] = 0
        mean, _, count, sum_val = stats.compute()
        value = sum_val.item() if metric == 'sum' else mean.item()
        pair_data[pair_key][label] = value
        pair_counts[pair_key] += count
    
    # First filter by count - keep only top percentage of pairs
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    cutoff_idx = max(1, int(len(sorted_pairs) * top_count_pct))
    top_pairs_by_count = sorted_pairs[:cutoff_idx]
    filtered_pairs = {pair: count for pair, count in top_pairs_by_count}
    
    # Calculate total for filtered pairs and sort
    pair_totals = {
        pair: sum(label_values.values()) 
        for pair, label_values in pair_data.items() 
        if pair in filtered_pairs
    }
    top_pairs = sorted(pair_totals.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Prepare data for plotting
    pairs = []
    for p, _ in top_pairs:
        current_token = p[0].replace('\n', '\\n')  # Escape newlines in tokens
        next_token = p[1].replace('\n', '\\n')
        pairs.append(f"{current_token}\n{next_token}")
    
    # Create data arrays for each label
    label_data = {label: [] for label in set(label for pair_dict in pair_data.values() for label in pair_dict.keys())}
    for pair, _ in top_pairs:
        pair_dict = pair_data[pair]
        for label in label_data:
            label_data[label].append(pair_dict.get(label, 0))
    
    # Create the stacked bar plot
    plt.figure(figsize=(20, 10))  # Make figure wider
    
    # Use a colormap for different labels
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(label_data)))
    
    bottom = np.zeros(len(pairs))
    bars = []
    for i, (label, values) in enumerate(label_data.items()):
        bar = plt.bar(range(len(pairs)), values, bottom=bottom, 
                     label=label, color=colors[i])
        bars.append(bar)
        bottom += np.array(values)
    
    # Adjust the text sizes
    plt.xticks(range(len(pairs)), pairs, rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=16)
    plt.title(f'Stacked KL Divergence by Token Pairs and Labels ({metric.capitalize()}, from top {int(top_count_pct*100)}% by count)', 
              fontsize=16)
    plt.xlabel('Token Pair', fontsize=16)
    plt.ylabel(f'{metric.capitalize()} KL Divergence', fontsize=16)
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', fontsize=16)
    
    plt.subplots_adjust(bottom=0.2)  # Add more space at bottom for labels
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../plots/stacked_kl_divergence_by_token_pairs_and_labels_{metric}_{deepseek_model_id}_{original_model_id}.png", 
                bbox_inches='tight', dpi=300)
    
    # Print the values with counts
    print(f"\nTop token pairs by label contributions ({metric}, filtered to top {int(top_count_pct*100)}% by count):")
    for pair, total in top_pairs:
        print(f"\n{pair[0]}, {pair[1]}: {total:.4f} total (count: {filtered_pairs[pair]})")
        pair_labels = pair_data[pair]
        for label, value in sorted(pair_labels.items(), key=lambda x: x[1], reverse=True):
            # Get the count from the original stats dictionary
            count = stats_dict[(pair[0], pair[1], label)].count
            print(f"  {label}: {value:.4f} (count: {int(count)})")

# Create the stacked bar plots for both metrics with default ignored categories
plot_stacked_token_pairs_by_label(stats_types["next_token_and_label"], metric='sum')
plot_stacked_token_pairs_by_label(stats_types["next_token_and_label"], metric='mean')

# Create plots without ignoring any categories
plot_stacked_token_pairs_by_label(stats_types["next_token_and_label"], metric='sum', ignore_categories=[])
plot_stacked_token_pairs_by_label(stats_types["next_token_and_label"], metric='mean', ignore_categories=[])

# %%
