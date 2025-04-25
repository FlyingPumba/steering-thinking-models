import dotenv
dotenv.load_dotenv("../.env")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from tqdm import tqdm
import gc
import time
import random
import torch.nn as nn
import openai
import anthropic
import os
from openai import OpenAI
import json
import re
import numpy as np

class LinearProbe(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
        
    def forward(self, x):
        return self.linear(x)


def chat(prompt, model="gpt-4.1", max_tokens=28000):

    model_provider = ""

    if model in ["gpt-4o", "gpt-4.1"]:
        model_provider = "openai"
        client = OpenAI()
    elif model in ["claude-3-opus", "claude-3-7-sonnet", "claude-3-5-haiku"]:
        model_provider = "anthropic"
        client = anthropic.Anthropic()
    elif model in ["deepseek-v3", "gemini-2-0-think", "gemini-2-0-flash", "deepseek-r1"]:
        model_provider = "openrouter"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            if model_provider == "openai":
                client = OpenAI(
                    organization="org-E6iEJQGSfb0SNHMw6NFT1Cmi",
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=1e-19,
                )
                return response.choices[0].message.content
            elif model_provider == "anthropic":
                model_mapping = {
                    "claude-3-opus": "claude-3-opus-latest",
                    "claude-3-7-sonnet": "claude-3-7-sonnet-latest",
                    "claude-3-5-haiku": "claude-3-5-haiku-latest"
                }

                if model == "claude-3-7-sonnet":
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        thinking = {
                            "type": "enabled",
                            "budget_tokens": max_tokens
                        },
                        max_tokens=max_tokens+1
                    )

                    thinking_response = response.content[0].thinking
                    answer_response = response.content[1].text

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"

                else:
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1e-19,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        max_tokens=max_tokens
                    )

                    return response.content[0].text
            elif model_provider == "openrouter":
                # Map model names to OpenRouter model IDs
                model_mapping = {
                    "deepseek-r1": "deepseek/deepseek-r1",
                    "deepseek-v3": "deepseek/deepseek-chat",
                    "gemini-2-0-think": "google/gemini-2.0-flash-thinking-exp:free",
                    "gemini-2-0-flash": "google/gemini-2.0-flash-001"
                }
                
                response = client.chat.completions.create(
                    model=model_mapping[model],
                    extra_body={},
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1e-19,
                    max_tokens=max_tokens
                )

                if hasattr(response.choices[0].message, "reasoning"):
                    thinking_response = response.choices[0].message.reasoning
                    answer_response = response.choices[0].message.content

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"
                else:
                    return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(20)

    return None



def generate_cluster_description(examples, model="gpt-4.1", n_trace_examples=0, model_name=None):
    """
    Generate a concise title and description for a cluster based on the top k examples.
    
    Args:
        examples (list): List of text examples from the cluster
        model (str): Model to use for generating the description
        n_trace_examples (int): Number of full reasoning trace examples to include in the prompt
        model_name (str): Name of the model whose responses should be loaded for trace examples
        
    Returns:
        tuple: (title, description) where both are strings
    """    
    # Prepare trace examples if requested
    trace_examples_text = ""
    if n_trace_examples > 0 and model_name is not None:
        try:
            # Get model identifier for file naming
            model_id = model_name.split('/')[-1].lower()
            responses_json_path = f"../train-steering-vectors/results/vars/responses_{model_id}.json"
            
            # Load responses
            with open(responses_json_path, 'r') as f:
                responses_data = json.load(f)
            
            # Select random examples
            trace_samples = random.sample(responses_data, min(n_trace_examples, len(responses_data)))
            
            # Extract thinking processes
            trace_examples = []
            for sample in trace_samples:
                if sample.get("thinking_process"):
                    trace_examples.append(sample["thinking_process"])
            
            if trace_examples:
                trace_examples_text = "Here are some full reasoning traces to help understand the context:\n'''\n"
                for i, trace in enumerate(trace_examples):
                    trace_examples_text += f"TRACE {i+1}:\n{trace}\n\n"
                trace_examples_text += "'''"
        except Exception as e:
            print(f"Error loading trace examples: {e}")
    
    # Create a prompt for the model
    prompt = f"""Analyze the following {len(examples)} sentences from an LLM reasoning trace. These sentences are grouped into a cluster based on their similar role or function in the reasoning process.

Your task is to identify the precise cognitive function these sentences serve in the reasoning process. Consider:
1. The reasoning strategy or cognitive operation being performed
2. Whether these sentences tend to appear in a specific position in reasoning (if applicable)

{trace_examples_text}

Examples:
'''
{chr(10).join([f"- {example}" for example in examples])}
'''

Look for:
- Shared reasoning strategies or cognitive mechanisms
- Common linguistic patterns or structures
- Positional attributes (only if clearly evident)
- Functional role within the overall reasoning process

Your response should be in this exact format:
Title: [concise title naming the specific reasoning function]
Description: [2-3 sentences explaining (1) what this function does, (2) what is INCLUDED and NOT INCLUDED in this category, and (3) position in reasoning if relevant]

Avoid overly general descriptions. Be precise enough that someone could reliably identify new examples of this reasoning function.
"""
        
    # Get the response from the model
    response = chat(prompt, model=model)
    
    # Parse the response to extract title and description
    title = "Unnamed Cluster"
    description = "No description available"
    
    title_match = re.search(r"Title:\s*(.*?)(?:\n|$)", response)
    if title_match:
        title = title_match.group(1).strip()
        
    desc_match = re.search(r"Description:\s*(.*?)(?:\n|$)", response)
    if desc_match:
        description = desc_match.group(1).strip()
    
    return title, description

def simplify_category_name(category_name):
    """
    Simplify a category name by extracting just the number if it matches 'Category N'
    or return the original name if not.
    """
    import re
    match = re.match(r'Category\s+(\d+)', category_name)
    if match:
        return match.group(1)
    return category_name

def completeness_autograder(sentences, categories, model="gpt-4.1"):
    """
    Autograder that evaluates if sentences belong to any of the provided categories.
    
    Args:
        sentences (list): List of sentences to evaluate
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
    
    Returns:
        dict: Statistics about category assignments including the fraction of sentences assigned/not assigned
    """
    # Format the categories into a readable list for the prompt
    categories_text = "\n\n".join([f"Category {cluster_id}: {title}\nDescription: {description}" 
                                  for cluster_id, title, description in categories])
    
    # Format the sentences into a numbered list
    sentences_text = "\n\n".join([f"Sentence {i}: {sentence}" for i, sentence in enumerate(sentences)])

    prompt = f"""# Task: Categorize Sentences of Reasoning Traces

You are an expert at categorizing the sentences of reasoning traces into predefined categories. Your task is to analyze each sentence and assign it to the most appropriate category based on the provided descriptions. If a sentence does not fit into any category, label it as "None".

## Categories:
{categories_text}

## Sentences to Categorize:
{sentences_text}

## Instructions:
1. For each sentence, carefully consider if it fits into one of the defined categories.
2. Assign exactly ONE category to each sentence if applicable, or "None" if it doesn't fit any category.
3. Provide your response in the exact format specified below.

## Response Format:
Your response must follow this exact JSON format:
```json
{{
  "categorizations": [
    {{
      "sentence_id": <sentence idx>,
      "assigned_category": "Category <category idx>" (not the title, just the category index) or "None",
      "explanation": "Brief explanation of your reasoning"
    }},
    ... (repeat for all sentences)
  ]
}}
```

Only include the JSON object in your response, with no additional text before or after.
"""
        
    # Call the chat API to get the categorization results
    response = chat(prompt, model=model)
    
    # Parse the response to extract the JSON
    try:
        import re
        import json
        
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find just the JSON object
            json_match = re.search(r'{\s*"categorizations":\s*\[[\s\S]*?\]\s*}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                # If all else fails, just try to use the entire response
                json_str = response
        
        result = json.loads(json_str)
        
        # Count the number of sentences assigned to each category and those not assigned
        total_sentences = len(sentences)
        assigned = 0
        not_assigned = 0
        category_counts = {str(cluster_id): 0 for cluster_id, _, _ in categories}
        category_counts["None"] = 0
        
        for item in result["categorizations"]:
            category = item["assigned_category"]
            if category == "None":
                not_assigned += 1
                category_counts["None"] += 1
            else:
                assigned += 1
                # Extract just the cluster ID from "Category N" format
                category_id = simplify_category_name(category)
                category_counts[category_id] = category_counts.get(category_id, 0) + 1
        
        # Calculate fractions
        assigned_fraction = assigned / total_sentences if total_sentences > 0 else 0
        not_assigned_fraction = not_assigned / total_sentences if total_sentences > 0 else 0
        
        return {
            "total_sentences": total_sentences,
            "assigned": assigned,
            "not_assigned": not_assigned,
            "assigned_fraction": assigned_fraction,
            "not_assigned_fraction": not_assigned_fraction,
            "category_counts": category_counts,
            "categorizations": result["categorizations"]
        }
    
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response}")
        return {
            "error": str(e),
            "total_sentences": len(sentences),
            "assigned": 0,
            "not_assigned": len(sentences),
            "assigned_fraction": 0,
            "not_assigned_fraction": 1.0,
            "raw_response": response
        }

def accuracy_autograder(sentences, categories, ground_truth_labels, model="gpt-4.1", n_autograder_examples=30):
    """
    Binary autograder that evaluates each cluster independently against examples from outside the cluster.
    
    Args:
        sentences (list): List of all sentences to potentially sample from
        categories (list): List of tuples where each tuple is (cluster_id, title, description)
        ground_truth_labels (list): List of cluster IDs (as strings) for each sentence in sentences
        model (str): Model to use for the autograding
        n_autograder_examples (int): Number of examples to sample from each cluster for testing
    
    Returns:
        dict: Metrics including precision, recall, accuracy and F1 score for each category
    """
    results = {}
    
    # Get a mapping from sentence index to cluster ID for easy lookup
    sentence_to_cluster = {i: label for i, label in enumerate(ground_truth_labels)}
    
    # For each category, evaluate independently
    for cluster_id, title, description in categories:
        cluster_id_str = str(cluster_id)
        
        # Find all examples in this cluster and not in this cluster
        in_cluster_indices = [i for i, label in enumerate(ground_truth_labels) if label == cluster_id_str]
        out_cluster_indices = [i for i, label in enumerate(ground_truth_labels) if label != cluster_id_str]
        
        # Get n_autograder_examples from the current cluster
        from_cluster_count = min(len(in_cluster_indices), n_autograder_examples)
        in_cluster_sample = random.sample(in_cluster_indices, from_cluster_count)
        
        # Get equal number of examples from outside the cluster
        from_outside_count = min(len(out_cluster_indices), n_autograder_examples)
        out_cluster_sample = random.sample(out_cluster_indices, from_outside_count)
        
        # Combine the samples and remember the ground truth
        test_indices = in_cluster_sample + out_cluster_sample
        test_sentences = [sentences[i] for i in test_indices]
        test_ground_truth = ["Yes" if i in in_cluster_sample else "No" for i in test_indices]
        
        # Shuffle to avoid position bias
        combined = list(zip(range(len(test_indices)), test_sentences, test_ground_truth))
        random.shuffle(combined)
        shuffled_indices, test_sentences, test_ground_truth = zip(*combined)
        
        # Create a prompt for binary classification
        prompt = f"""# Task: Binary Classification of Reasoning Sentences

You are an expert at analyzing reasoning traces. I'll provide a description of a specific reasoning function or pattern, along with several example sentences. Your task is to determine whether each sentence belongs to this category or not.

## Category Description:
Title: {title}
Description: {description}

## Sentences to Classify:
{chr(10).join([f"Sentence {i}: {sentence}" for i, sentence in enumerate(test_sentences)])}

## Instructions:
1. For each sentence, determine if it belongs to the described category.
2. Respond with "Yes" if it belongs to the category, or "No" if it does not.
3. Provide your response in the exact format specified below.

## Response Format:
Your response must follow this exact JSON format:
```json
{{
  "classifications": [
    {{
      "sentence_id": <sentence idx>,
      "belongs_to_category": "Yes" or "No",
      "explanation": "Brief explanation of your reasoning"
    }},
    ... (repeat for all sentences)
  ]
}}
```

Only include the JSON object in your response, with no additional text before or after.
"""
        
        # Call the chat API to get the classification results
        response = chat(prompt, model=model)
        
        # Parse the response to extract the JSON
        try:
            import re
            import json
            
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find just the JSON object
                json_match = re.search(r'{\s*"classifications":\s*\[[\s\S]*?\]\s*}', response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # If all else fails, just try to use the entire response
                    json_str = response
            
            result = json.loads(json_str)
            
            # Compute metrics for this cluster
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            predictions = []
            
            for item in result["classifications"]:
                sentence_idx = item["sentence_id"]
                belongs = item["belongs_to_category"]
                predictions.append(belongs)
                
                true_label = test_ground_truth[sentence_idx]
                
                if belongs == "Yes" and true_label == "Yes":
                    true_positives += 1
                elif belongs == "Yes" and true_label == "No":
                    false_positives += 1
                elif belongs == "No" and true_label == "Yes":
                    false_negatives += 1
                elif belongs == "No" and true_label == "No":
                    true_negatives += 1
            
            # Calculate metrics
            accuracy = (true_positives + true_negatives) / len(test_sentences) if test_sentences else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[cluster_id_str] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "predictions": predictions,
                "classifications": result["classifications"]
            }
        
        except Exception as e:
            print(f"Error in accuracy autograder for cluster {cluster_id}: {e}")
            print(f"Raw response: {response}")
            results[cluster_id_str] = {
                "error": str(e),
                "raw_response": response
            }
    
    # Calculate overall averages across all clusters
    if results:
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            avg_accuracy = sum(r["accuracy"] for r in valid_results.values()) / len(valid_results)
            avg_precision = sum(r["precision"] for r in valid_results.values()) / len(valid_results)
            avg_recall = sum(r["recall"] for r in valid_results.values()) / len(valid_results)
            avg_f1 = sum(r["f1"] for r in valid_results.values()) / len(valid_results)
            
            results["avg"] = {
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
    
    return results


def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token

def process_saved_responses(model_name, n_examples, model, tokenizer, layer):
    """Load and process saved responses to get activations"""
    print(f"Processing saved responses for {model_name}...")
    
    # Load model and tokenizer
    model_id = model_name.split('/')[-1].lower()
    responses_json_path = f"../train-steering-vectors/results/vars/responses_{model_id}.json"
    
    print(f"Loading responses from {responses_json_path}...")
    try:
        with open(responses_json_path, 'r') as f:
            responses_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {responses_json_path} not found.")
        return [], []
    
    # Limit to n_examples
    import random
    random.shuffle(responses_data)
    responses_data = responses_data[:n_examples]
        
    # Extract text segments and their activations
    all_activations = []
    all_texts = []
    
    overall_running_mean = torch.zeros(1, model.config.hidden_size)
    overall_running_count = 0

    print("Extracting activations for sentences...")
    from tqdm import tqdm
    for response_data in tqdm(responses_data):
        if not response_data.get("thinking_process"):
            continue
            
        # Get the thinking process text
        thinking_text = response_data["thinking_process"]
        full_response = response_data["full_response"]
        
        # Split into sentences using regex
        sentences = re.split(r'[.!?;]', thinking_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s.split()) >= 5]
        
        # Encode the full response to get input_ids
        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        
        # Get layer activations
        with model.trace(input_ids) as tracer:
            layer_outputs = model.model.layers[layer].output[0].save()
        
        # Convert layer outputs to numpy arrays
        layer_outputs = layer_outputs.detach().to(torch.float32)
        
        # Create character to token mapping
        char_to_token = get_char_to_token_map(full_response, tokenizer)
        
        # Process each sentence
        min_token_start = float('inf')
        max_token_end = -float('inf')
        for sentence in sentences:
            # Find this sentence in the original text
            text_pos = full_response.find(sentence)
            if text_pos >= 0:
                # Get start and end token positions
                token_start = char_to_token.get(text_pos, None)
                token_end = char_to_token.get(text_pos + len(sentence), None)
                
                if token_start is not None and token_end is not None and token_start < token_end:
                    if token_start < min_token_start:
                        min_token_start = token_start
                    if token_end > max_token_end:
                        max_token_end = token_end

                    # Extract activations for this segment
                    segment_activations = layer_outputs[:, token_start-1:token_end, :].mean(dim=1).cpu()  # Average over tokens
                                        
                    # Save the result
                    all_activations.append(segment_activations)  # Store as torch tensor
                    all_texts.append(sentence)
    
        if min_token_start < layer_outputs.shape[1] and max_token_end > 0:
            vector = layer_outputs[:,min_token_start:max_token_end,:].mean(dim=1).cpu()
            overall_running_mean = overall_running_mean + (vector - overall_running_mean) / (overall_running_count + 1)
            overall_running_count += 1

    return all_activations, all_texts, overall_running_mean

def load_model_and_vectors(device="cuda:0", load_in_8bit=False, compute_features=True, normalize_features=True, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", base_model_name=None):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        load_in_8bit (bool): If True, load the model in 8-bit mode
        compute_features (bool): If True, compute and return feature vectors by subtracting overall mean
        normalize_features (bool): If True, normalize the feature vectors
        return_steering_vector_set (bool): If True, return the steering vector set
        model_name (str): Name/path of the model to load
        base_model_name (str): Name/path of the base model to load
    """
    model = LanguageModel(model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    if base_model_name is not None:
        base_model = LanguageModel(base_model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
    
        base_model.generation_config.temperature=None
        base_model.generation_config.top_p=None
        base_model.generation_config.do_sample=False
        
        base_tokenizer = base_model.tokenizer
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    
    # go into directory of this file
    mean_vectors_dict = torch.load(f"../train-steering-vectors/results/vars/mean_vectors_{model_id}.pt")

    if compute_features:
        # Compute feature vectors by subtracting overall mean
        feature_vectors = {}
        feature_vectors["overall"] = mean_vectors_dict["overall"]['mean']
        
        for label in mean_vectors_dict:

            if label != 'overall':
                feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']

            if normalize_features:
                for label in feature_vectors:
                    feature_vectors[label] = feature_vectors[label] * (feature_vectors["overall"].norm(dim=-1, keepdim=True) / feature_vectors[label].norm(dim=-1, keepdim=True))

    if base_model_name is not None and compute_features:
        return model, tokenizer, base_model, base_tokenizer, feature_vectors
    elif base_model_name is not None and not compute_features:
        return model, tokenizer, base_model, base_tokenizer, mean_vectors_dict
    elif base_model_name is None and compute_features:
        return model, tokenizer, feature_vectors
    else:
        return model, tokenizer, mean_vectors_dict

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, steering_config, steer_positive=False):
    """
    Generate text while removing or adding projections of specific features.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors containing steering_vector_set
        steer_positive: If True, steer towards the label, if False steer away
    """
    model_layers = model.model.layers

    with model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if feature_vectors is not None:       
            vector_layer = steering_config[label]["vector_layer"]
            pos_layers = steering_config[label]["pos_layers"]
            neg_layers = steering_config[label]["neg_layers"]
            coefficient = steering_config[label]["pos_coefficient"] if steer_positive else steering_config[label]["neg_coefficient"]
     

            if steer_positive:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in pos_layers:         
                    model.model.layers[layer_idx].output[0][:, :] += coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
            else:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in neg_layers:         
                    model.model.layers[layer_idx].output[0][:, :] -= coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
        
        outputs = model.generator.output.save()
                    
    return outputs

def get_random_distinct_colors(labels):
    """
    Generate random distinct ANSI colors for each label.
    
    Args:
        labels: List of label names
        
    Returns:
        Dictionary mapping labels to ANSI color codes
    """
    import random
    
    # List of distinct ANSI colors (excluding black, white, and hard-to-see colors)
    # Format is "\033[COLORm" where COLOR is a number between 31-96
    distinct_colors = [
        "\033[31m",  # Red
        "\033[32m",  # Green
        "\033[33m",  # Yellow
        "\033[34m",  # Blue
        "\033[35m",  # Magenta
        "\033[36m",  # Cyan
        "\033[91m",  # Bright Red
        "\033[92m",  # Bright Green
        "\033[93m",  # Bright Yellow
        "\033[94m",  # Bright Blue
        "\033[95m",  # Bright Magenta
        "\033[96m",  # Bright Cyan
    ]
    
    # Shuffle the colors to randomize them
    random.shuffle(distinct_colors)
    
    # Ensure we have enough colors
    if len(labels) > len(distinct_colors):
        # If we need more colors, create additional ones with random RGB values
        additional_needed = len(labels) - len(distinct_colors)
        for _ in range(additional_needed):
            # Generate random RGB foreground color (38;2;r;g;b)
            r, g, b = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
            # Ensure colors are distinct by checking minimum distance from existing colors
            # (simplified approach)
            distinct_colors.append(f"\033[38;2;{r};{g};{b}m")
    
    # Assign colors to labels
    label_colors = {}
    for i, label in enumerate(labels):
        label_colors[label] = distinct_colors[i % len(distinct_colors)]
    
    return label_colors

# Create NumpyEncoder for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


#  problem-framing
#  analytical-decomposition
#  structural-decomposition
#  possibility-checking
#  calculation-computation
#  hypothesis-generation
#  generating-additional-considerations
#  logical-structure-testing

steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "backtracking": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "backtracking": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "backtracking": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 44, "pos_layers": [44], "neg_layers": [44], "pos_coefficient": 1, "neg_coefficient": 1},
    }
}
