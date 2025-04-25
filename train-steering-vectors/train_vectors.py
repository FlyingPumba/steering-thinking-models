# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer
import torch
import re
from nnsight import NNsight
from collections import defaultdict
from messages import messages
from tqdm import tqdm
import os
from messages import labels
import random
import json
import utils
import math
import gc
# Parse arguments
parser = argparse.ArgumentParser(description="Train steering vectors for model reasoning")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--save_every", type=int, default=1, 
                    help="Save checkpoints every n examples")
parser.add_argument("--load_from_json", action="store_true", default=False,
                    help="Load responses from JSON instead of generating new ones")
parser.add_argument("--update_annotation", action="store_true", default=False,
                    help="Only update annotations in the existing JSON file without recalculating vectors")
parser.add_argument("--max_tokens", type=int, default=500,
                    help="Maximum number of tokens to generate")
parser.add_argument("--n_samples", type=int, default=100,
                    help="Number of samples to process")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for processing messages")
args, _ = parser.parse_known_args()

# python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n_samples 500 --max_tokens 1000 --batch_size 4 --save_every 1 --load_from_json --update_annotation

# %%
def get_batched_message_ids(tokenizer, messages_list, apply_chat_template=True):
    if apply_chat_template:
        tokenized_messages = [tokenizer.apply_chat_template([msg], add_generation_prompt=True, return_tensors="pt")[0] for msg in messages_list]
    else:
        tokenized_messages = [tokenizer.encode(msg, return_tensors="pt")[0] for msg in messages_list]
    max_token_length = max([len(tokens) for tokens in tokenized_messages])
    
    # Create padded input_ids and attention masks
    input_ids = []
    attention_masks = []
    
    for tokens in tokenized_messages:
        pad_length = max_token_length - len(tokens)
        # Left padding
        padded_ids = torch.cat([
            torch.ones(pad_length, dtype=torch.long) * tokenizer.pad_token_id,
            tokens
        ])
        # Mask: 0 for padding, 1 for content
        mask = torch.cat([
            torch.zeros(pad_length, dtype=torch.long),
            torch.ones(len(tokens), dtype=torch.long)
        ])
        
        input_ids.append(padded_ids)
        attention_masks.append(mask)
    
    input_ids = torch.stack(input_ids).to("cuda")
    attention_masks = torch.stack(attention_masks).to("cuda")
    
    return input_ids, attention_masks

def process_saved_responses_batch(responses_list, tokenizer, model):
    """Get layer activations for a batch of saved responses without generation"""
    tokenized_responses, attention_masks = get_batched_message_ids(tokenizer, responses_list, apply_chat_template=False)
    
    # Process the inputs through the model to get activations
    layer_outputs = []
    with model.trace(tokenized_responses) as tracer:
        
        # Capture layer outputs
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = [x.value.cpu().detach().to(torch.float32) for x in layer_outputs]

    batch_layer_outputs = []
    
    for batch_idx in range(len(responses_list)):
        # get length of padding tokens
        padding_length = (attention_masks[batch_idx].squeeze() == 0).sum().item()
        
        # Slice out just the non-padded activations for this example across all layers
        example_outputs = torch.stack([
            layer_output[batch_idx][padding_length:] 
            for layer_output in layer_outputs
        ])
        
        batch_layer_outputs.append(example_outputs)
    
    return batch_layer_outputs

def process_model_output_batch(messages_batch, tokenizer, model):
    """Get model output and layer activations for a batch of messages"""
    tokenized_messages, attention_masks = get_batched_message_ids(tokenizer, messages_batch, apply_chat_template=True)
    
    with model.generate(
        tokenized_messages,
        max_new_tokens=args.max_tokens,
        pad_token_id=tokenizer.eos_token_id
    ) as tracer:
        outputs = model.generator.output.save()

    # Process the whole batch at once
    layer_outputs = []
    with model.trace(outputs):
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = [x.value.cpu().detach().to(torch.float32) for x in layer_outputs]

    batch_layer_outputs = []
    
    for batch_idx in range(len(messages_batch)):
        # get length of padding tokens
        padding_length = (attention_masks[batch_idx].squeeze() == 0).sum().item()
        
        # Slice out just the non-padded activations for this example across all layers
        example_outputs = torch.stack([
            layer_output[batch_idx][padding_length:] 
            for layer_output in layer_outputs
        ])
        
        batch_layer_outputs.append(example_outputs)
    
    return outputs, batch_layer_outputs

def extract_thinking_process(response):
    """Extract thinking process from response"""
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    return response[think_start:think_end].strip()

def get_label_positions(annotated_response, response_text, tokenizer):
    """Parse annotations and find token positions for each label more accurately"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments properly
    pattern = r'\["([\w-]+)"\](.*?)(?=\["[\w-]+"\]|$)'
    matches = list(re.finditer(pattern, annotated_response, re.DOTALL))
    
    # Create character to token mapping once
    char_to_token = get_char_to_token_map(response_text, tokenizer)
    
    for match in matches:
        labels_str = match.group(1)
        labels = [label.strip() for label in labels_str.split(',')]
        
        # Skip end-section markers
        if "end-section" in labels:
            continue
            
        text = match.group(2).replace('["<end-section>"]', "").strip()
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        text_pos = response_text.find(text)
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text), None)

            if token_start == token_end:
                continue
            
            # If we found valid token positions
            if token_start is not None and token_end is not None:
                for label in labels:
                    if label not in label_positions:
                        label_positions[label] = []
                    label_positions[label].append((token_start, token_end))
    
    return label_positions

def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token

def update_mean_vectors(mean_vectors, layer_outputs, label_positions, index):
    """Update mean vectors for overall and individual labels"""
    # Calculate overall thinking section boundaries
    all_positions = [pos for positions in label_positions.values() for pos in positions]
    if all_positions:
        min_pos = min(start for start, _ in all_positions)
        max_pos = max(end for _, end in all_positions)
        
        # Update overall mean
        overall_vectors = layer_outputs[:, min_pos:max_pos].mean(dim=1)
        current_count = mean_vectors['overall']['count']
        current_mean = mean_vectors['overall']['mean']
        mean_vectors['overall']['mean'] = current_mean + (overall_vectors - current_mean) / (current_count + 1)
        mean_vectors['overall']['count'] += 1

        if torch.isnan(mean_vectors['overall']['mean']).any():
            print(f"NaN in mean_vectors['overall']['mean'] at index {index}")
    
    # Update individual labels
    for label, positions in label_positions.items():
        for position in positions:
            start, end = position
            vectors = layer_outputs[:, start-1:end-1].mean(dim=1)
            if torch.isnan(vectors).any():
                print(f"NaN in mean_vectors['{label}']['mean'] at index {index}")
                print(f"Layer outputs: {layer_outputs[:, start-1:min(end-1, start+2)]}")
                print(f"Layer outputs shape: {layer_outputs.shape}")
                print(f"Positions: {positions}")
                print(f"Index: {index}")
                print(f"Label: {label}")
                print(f"Start: {start}")
                print(f"End: {end}")
                print(f"Vectors: {vectors}")
                print(f"Current count: {current_count}")
                print(f"Current mean: {current_mean}")
                print(f"Mean vectors: {mean_vectors[label]['mean']}")
                
                continue
            
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            mean_vectors[label]['mean'] = current_mean + (vectors - current_mean) / (current_count + 1)
            mean_vectors[label]['count'] += 1

def process_batch_annotations(thinking_processes):
    """Get annotations for a batch of thinking processes"""
    # This function needs to call API sequentially as it can't be batched
    annotated_responses = []
    for thinking in thinking_processes:
        annotated_response = utils.chat(f"""
        Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

        Available labels:
        0. initializing -> The model is rephrasing the given task and states initial thoughts.
        1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
        2. adding-knowledge -> The model is enriching the current approach with recalled facts.
        3. example-testing -> The model generates examples to test its current approach.
        4. uncertainty-estimation -> The model is stating its own uncertainty.
        5. backtracking -> The model decides to change its approach.

        The reasoning chain to analyze:
        {thinking}

        Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
        """)
        annotated_responses.append(annotated_response)
    
    return annotated_responses


def process_message_batch(messages_batch, tokenizer, model, mean_vectors, batch_indices, get_annotation=True):
    """Process a batch of messages and update mean vectors"""
    outputs, batch_layer_outputs = process_model_output_batch(messages_batch, tokenizer, model)
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    thinking_processes = [extract_thinking_process(response) for response in responses]
    
    annotated_responses = process_batch_annotations(thinking_processes) if get_annotation else [None] * len(messages_batch)
    
    batch_data = []
    for i, (message, response, thinking, annotated) in enumerate(zip(messages_batch, responses, thinking_processes, annotated_responses)):
        batch_data.append({
            "original_message": message,
            "full_response": response,
            "thinking_process": thinking,
            "annotated_thinking": annotated
        })
        
        if annotated:
            # No need to adjust for padding here since we've already sliced it out
            label_positions = get_label_positions(annotated, tokenizer.decode(outputs[i]), tokenizer)
            update_mean_vectors(mean_vectors, batch_layer_outputs[i], label_positions, batch_indices[i])
    
    return batch_data

# %% Main execution
model_name = args.model

# Load model using utils function
print(f"Loading model {model_name}...")
model, tokenizer, _ = utils.load_model_and_vectors(compute_features=False, model_name=model_name, load_in_8bit=args.load_in_8bit)

mean_vectors = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
    'count': 0
})

# %%
# Create directories
os.makedirs('results/vars', exist_ok=True)

save_every = args.save_every
save_path = f"results/vars/mean_vectors_{model_name.split('/')[-1].lower()}.pt"

load_from_json = args.load_from_json
update_annotation = args.update_annotation
responses_json_path = f"results/vars/responses_{model_name.split('/')[-1].lower()}.json"

responses_data = []
random.seed(args.seed)

if update_annotation and os.path.exists(responses_json_path):
    print(f"Loading existing responses from {responses_json_path} to update annotations and vectors")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    
    # Process in batches to update annotations and vectors
    num_batches = math.ceil(min(len(responses_data), args.n_samples) / args.batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Updating annotations and vectors"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, min(len(responses_data), args.n_samples))
        
        batch_responses = responses_data[start_idx:end_idx]
        thinking_processes = [data["thinking_process"] for data in batch_responses]
        batch_full_responses = [data["full_response"] for data in batch_responses]
        batch_indices = list(range(start_idx, end_idx))
        
        # Update annotations
        annotated_responses = process_batch_annotations(thinking_processes)
        
        # Update annotation fields in the JSON
        for i, (response_data, annotated) in enumerate(zip(batch_responses, annotated_responses)):
            responses_data[start_idx + i]["annotated_thinking"] = annotated
        
        # Process saved responses to calculate vectors
        batch_layer_outputs = process_saved_responses_batch(batch_full_responses, tokenizer, model)
        
        # Update vectors based on new annotations
        for i, (response_data, layer_outputs) in enumerate(zip(batch_responses, batch_layer_outputs)):
            if annotated_responses[i]:  # Use the new annotations
                label_positions = get_label_positions(annotated_responses[i], response_data["full_response"], tokenizer)
                update_mean_vectors(mean_vectors, layer_outputs, label_positions, batch_indices[i])
                
                del layer_outputs
        
        if batch_idx % save_every == 0:
            # Save updated JSON
            with open(responses_json_path, "w") as f:
                json.dump(responses_data, f, indent=2)
            # Save updated vectors
            save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
            torch.save(save_dict, save_path)

        del batch_layer_outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save final results
    with open(responses_json_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
    torch.save(save_dict, save_path)
    print("Saved updated annotations and vectors")
    
elif load_from_json and os.path.exists(responses_json_path):
    print(f"Loading existing responses from {responses_json_path} to update vectors")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    random.shuffle(responses_data)
    
    # Process in batches
    num_batches = math.ceil(min(len(responses_data), args.n_samples) / args.batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Processing saved response batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, min(len(responses_data), args.n_samples))
        
        batch_responses = responses_data[start_idx:end_idx]
        batch_full_responses = [data["full_response"] for data in batch_responses]
        batch_indices = list(range(start_idx, end_idx))
        
        # Process saved responses without regenerating
        batch_layer_outputs = process_saved_responses_batch(batch_full_responses, tokenizer, model)
        
        for i, (response_data, layer_outputs) in enumerate(zip(batch_responses, batch_layer_outputs)):
            if response_data["annotated_thinking"]:
                label_positions = get_label_positions(response_data["annotated_thinking"], response_data["full_response"], tokenizer)
                update_mean_vectors(mean_vectors, layer_outputs, label_positions, batch_indices[i])

                del layer_outputs
        
        if batch_idx % save_every == 0:
            save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
            torch.save(save_dict, save_path)

        del batch_layer_outputs
        torch.cuda.empty_cache()
        gc.collect()
else:
    print(f"Processing {args.n_samples} messages to generate new vectors")
    random.shuffle(messages)
    messages = messages[:args.n_samples]
    num_batches = math.ceil(len(messages) / args.batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Processing message batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(messages))
        
        batch_messages = messages[start_idx:end_idx]
        batch_indices = list(range(start_idx, end_idx))
        
        batch_data = process_message_batch(batch_messages, tokenizer, model, mean_vectors, batch_indices)
        responses_data.extend(batch_data)
        
        if batch_idx % save_every == 0:
            save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
            torch.save(save_dict, save_path)
            with open(responses_json_path, "w") as f:
                json.dump(responses_data, f, indent=2)

# Save final results
with open(responses_json_path, "w") as f:
    json.dump(responses_data, f, indent=2)
print("Saved final responses data")

save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final mean vectors")