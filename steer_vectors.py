import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from jaxtyping import Float
import config
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("DEVICE", device)

def create_contrastive_pairs(
        behavioral_prompts: Dict[str, List[str]], 
        num_pairs_per_category: int = 100
        ) -> Dict[str, Dict[str, List[str]]]:
    contrastive_pairs = {}
    print("Creating contrastive pairs for refusal....")
    refusal_positive = behavioral_prompts['refusal'][:num_pairs_per_category]
    refusal_negative = behavioral_prompts['helpfulness'][:num_pairs_per_category]
    contrastive_pairs['refusal'] = {
        'positive': refusal_positive,
        'negative': refusal_negative
    }

    print("Creating contrastive pairs for roleplay...")
    roleplay_positive = behavioral_prompts['roleplay'][:num_pairs_per_category]
    roleplay_negative = []
    for prompt in roleplay_positive:
        neutral = prompt
        for trigger in ['You are a ', 'Act as a ', 'Pretend you\'re a ', 'Imagine you are a ', 'Roleplay as a ']:
            if trigger in neutral:
                parts = neutral.split('. ')
                if len(parts) > 1:
                    neutral = parts[-1] 
                break
        roleplay_negative.append(neutral)
    contrastive_pairs['roleplay'] = {
        'positive': roleplay_positive,
        'negative': roleplay_negative
    }

    print("reating contrastive pairs for uncertainty...")
    uncertainty_positive = behavioral_prompts['uncertainty'][:num_pairs_per_category]
    uncertainty_negative = [
        "What is the capital of France?",
        "When did World War II end?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
    ] * (num_pairs_per_category // 5 + 1)
    uncertainty_negative = uncertainty_negative[:num_pairs_per_category]
    contrastive_pairs['uncertainty'] = {
        'positive': uncertainty_positive,
        'negative': uncertainty_negative
    }

    print("Creating contrastive pairs for helpfulness...")
    helpfulness_positive = behavioral_prompts['helpfulness'][:num_pairs_per_category]
    helpfulness_negative = behavioral_prompts['refusal'][:num_pairs_per_category]
    contrastive_pairs['helpfulness'] = {
        'positive': helpfulness_positive,
        'negative': helpfulness_negative
    }

    print("Creating contrastive pairs for format...")
    format_positive = behavioral_prompts['format'][:num_pairs_per_category]
    format_negative = [
        "Tell me about climate change.",
        "Explain artificial intelligence.",
        "Describe renewable energy.",
        "What is machine learning?",
        "Discuss space exploration.",
    ] * (num_pairs_per_category // 5 + 1)
    format_negative = format_negative[:num_pairs_per_category]
    contrastive_pairs['format'] = {
        'positive': format_positive,
        'negative': format_negative
    }

    print("Done")
    return contrastive_pairs

@torch.no_grad() 
def extract_activations(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        prompts: List[str], 
        layer_idx: int, 
        device: torch.device, 
        batch_size: int = 8
        ) -> Float[torch.Tensor, "num_prompts hidden_dim"]:
    all_activations = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        encoded = tokenizer(batch_prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        layer_hidden_states = outputs.hidden_states[layer_idx + 1] 
        mean_activations = layer_hidden_states.mean(dim=1)

        all_activations.append(mean_activations.cpu())
        del outputs, layer_hidden_states, input_ids, attention_mask
        torch.cuda.empty_cache()
    return torch.cat(all_activations, dim=0) 

@torch.no_grad() 
def compute_steering_vectors(
    contrastive_pairs: Dict[str, Dict[str, List[str]]], 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    layer_idx: int, 
    device: torch.device
    ) -> Dict[str, Float[torch.Tensor, "hidden_dim"]]:
    """
    steering vector = mean(positive_activations) - mean(negative_activations)
    """
    steering_vectors = {}
    for category in contrastive_pairs.keys():
        positive_prompts = contrastive_pairs[category]['positive']
        negative_prompts = contrastive_pairs[category]['negative']
        positive_acts = extract_activations(model, tokenizer, positive_prompts, layer_idx, device, batch_size=4)
        negative_acts = extract_activations(model, tokenizer, negative_prompts, layer_idx, device, batch_size=4)

        positive_mean = positive_acts.mean(dim=0)  
        negative_mean = negative_acts.mean(dim=0)
        steering_vector = positive_mean - negative_mean 

        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
        steering_vectors[category] = steering_vector.float()

    return steering_vectors

def main():
    chat_model = AutoModelForCausalLM.from_pretrained(config.chat_model_id, torch_dtype=torch.float16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(config.chat_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    with open(config.behavioral_prompts_filepath, 'r') as f:
        output = json.load(f)

    all_behavioral_prompts = output['prompts']
    contrastive_pairs = create_contrastive_pairs(all_behavioral_prompts, num_pairs_per_category=100)

    steering_vectors = compute_steering_vectors(contrastive_pairs, chat_model, tokenizer, config.target_layer, device)
    torch.save(steering_vectors, config.steering_vectors_filepath)

if __name__ == "__main__":
    main()