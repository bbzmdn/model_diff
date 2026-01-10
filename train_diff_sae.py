import torch
import torch.nn as nn
from datasets import load_dataset, Dataset, IterableDataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import datasets
import numpy as np
import glob
from tqdm import tqdm
import os
import gc
from diff_sae import DiffSAE
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
import config

def extract_diffs_streaming(
    dataset: Union[Dataset, IterableDataset],
    base_model: PreTrainedModel,
    chat_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int = 100_000_000,
    max_length: int = 512,
    batch_size: int = 16,
    layer_idx: int = 13,
    save_frequency: int = 5_000_000,
    output_dir: str = "diff_activations",
) -> int:
  
    os.makedirs(output_dir, exist_ok=True)
    
    all_diffs = []
    total_tokens = 0
    chunk_idx = 0
    texts_in_batch = []
    
    for item in tqdm(dataset):
        text = item['text']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        if n_tokens < 50:
            continue
        if n_tokens > max_length:
            text = tokenizer.decode(tokens[:max_length])
            n_tokens = max_length
        
        texts_in_batch.append(text)
        total_tokens += n_tokens
       
        if len(texts_in_batch) >= batch_size:
            inputs = tokenizer(
                texts_in_batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to('cuda')
            
            with torch.no_grad():
                base_outputs = base_model(**inputs, output_hidden_states=True)
                chat_outputs = chat_model(**inputs, output_hidden_states=True)
                
                base_acts = base_outputs.hidden_states[layer_idx]
                chat_acts = chat_outputs.hidden_states[layer_idx]
                
                base_mean = base_acts.mean(dim=1)
                chat_mean = chat_acts.mean(dim=1)
                
                diffs = chat_mean - base_mean  
                
                all_diffs.append(diffs.cpu().half())

            texts_in_batch = []
            del inputs, base_outputs, chat_outputs, base_acts, chat_acts
            torch.cuda.empty_cache()
        if total_tokens > 0 and total_tokens % save_frequency < batch_size * max_length:
            if len(all_diffs) > 0:
                diffs_tensor = torch.cat(all_diffs, dim=0)
                torch.save(
                    diffs_tensor,
                    f'{output_dir}/diffs_chunk_{chunk_idx:03d}.pt'
                )
                
                all_diffs = []
                del diffs_tensor
                gc.collect()
                chunk_idx += 1
      
        if total_tokens >= target_tokens:
            break
    
    if len(texts_in_batch) > 0:
        inputs = tokenizer(
            texts_in_batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to('cuda')
        
        with torch.no_grad():
            base_outputs = base_model(**inputs, output_hidden_states=True)
            chat_outputs = chat_model(**inputs, output_hidden_states=True)
            
            base_mean = base_outputs.hidden_states[layer_idx].mean(dim=1)
            chat_mean = chat_outputs.hidden_states[layer_idx].mean(dim=1)
            diffs = chat_mean - base_mean
            
            all_diffs.append(diffs.cpu().half())
        
        del inputs, base_outputs, chat_outputs
        torch.cuda.empty_cache()
    
    if len(all_diffs) > 0:
        diffs_tensor = torch.cat(all_diffs, dim=0)
        torch.save(
            diffs_tensor,
            f'{output_dir}/diffs_chunk_{chunk_idx:03d}.pt'
        )
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total chunks: {chunk_idx + 1}")


def train_diffsae_chunked(
    model: DiffSAE,
    chunk_dir: str = "diff_activations",
    num_chunks: int = 20,
    num_epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-4,
    aux_coef: float = 1 / 32,
    log_interval: int = 100,
    save_dir: str = "diffsae_checkpoints",
    device: torch.device = torch.device("cpu"),
) -> Tuple[DiffSAE, List[float]]:
    
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    chunk_files = sorted(glob.glob(f"{chunk_dir}/diffs_chunk_*.pt"))
    if len(chunk_files) == 0:
        raise ValueError(f"No chunk files found in {chunk_dir}")
    
    print(f"Starting diff-SAE training on chunks")
    print(f"Num chunks: {len(chunk_files)}")
    
    global_step = 0
    best_loss = float('inf')
    all_epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_aux_loss = 0
        epoch_l0 = 0
        total_batches = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print()
        
        for chunk_idx, chunk_path in enumerate(chunk_files):
            print(f"\nLoading chunk {chunk_idx+1}/{len(chunk_files)}: {os.path.basename(chunk_path)}")
            chunk_data = torch.load(chunk_path, map_location='cpu')
            if chunk_data.dtype == torch.float16:
                chunk_data = chunk_data.float()
            
            dataset = TensorDataset(chunk_data)
            chunk_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
            chunk_loss = 0
            chunk_aux_loss = 0
            chunk_l0 = 0
            
            pbar = tqdm(chunk_loader, desc=f"Chunk {chunk_idx+1}/{len(chunk_files)}", leave=False) 
            for batch_idx, (batch,) in enumerate(pbar):
                x = batch.to(device)
                x_recon, latents, aux_loss = model(x, return_aux=True)
                recon_loss = nn.functional.mse_loss(x_recon, x)
                loss = recon_loss + aux_coef * aux_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #Normalize decoder weights
                with torch.no_grad():
                    model.decoder.weight.data = nn.functional.normalize(
                        model.decoder.weight.data, dim=0
                    )
                l0 = (latents != 0).float().sum(dim=-1).mean().item()
                chunk_loss += recon_loss.item()
                chunk_aux_loss += aux_loss.item()
                chunk_l0 += l0
                
                epoch_loss += recon_loss.item()
                epoch_aux_loss += aux_loss.item()
                epoch_l0 += l0
                total_batches += 1
                
                if global_step % log_interval == 0:
                    pbar.set_postfix({'loss': f'{recon_loss.item():.4f}',
                                      'aux': f'{aux_loss.item():.4f}',
                                      'L0': f'{l0:.1f}'})
                
                global_step += 1
            avg_chunk_loss = chunk_loss / len(chunk_loader)
            avg_chunk_aux = chunk_aux_loss / len(chunk_loader)
            avg_chunk_l0 = chunk_l0 / len(chunk_loader)
            
            print(f"Chunk {chunk_idx+1} complete:")
            print(f"Loss: {avg_chunk_loss:.6f} Aux: {avg_chunk_aux:.6f}  L0: {avg_chunk_l0:.2f}")
            
            del chunk_data, dataset, chunk_loader
            torch.cuda.empty_cache()
            gc.collect()

        avg_loss = epoch_loss / total_batches
        avg_aux_loss = epoch_aux_loss / total_batches
        avg_l0 = epoch_l0 / total_batches
        all_epoch_losses.append(avg_loss)
        print()
        print(f"Epoch {epoch+1}")
        print(f"Epoch reconstruction Loss: {avg_loss:.6f}")
        print(f"Epoch auxiliary Loss: {avg_aux_loss:.6f}")
        print(f"Epoch L0 Sparsity: {avg_l0:.2f}")
        
        # check if loss plateaued
        if len(all_epoch_losses) >= 3:
            recent_losses = all_epoch_losses[-3:]
            loss_change = (max(recent_losses) - min(recent_losses)) / max(recent_losses)
            print(f"  Recent loss variation: {loss_change:.4f}")
            if loss_change < 0.001:
                print(f"Loss plateaued (change < 0.1%)")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(save_dir, "best_diffsae.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'aux_loss': avg_aux_loss,
                'l0': avg_l0,
                'all_losses': all_epoch_losses,
            }, checkpoint_path)
       
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"diffsae_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'aux_loss': avg_aux_loss,
                'l0': avg_l0,
                'all_losses': all_epoch_losses,
            }, checkpoint_path)
   
    print("Done")

    # plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_epoch_losses) + 1), all_epoch_losses, 'b-', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.title('Diff-SAE Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    
    return model, all_epoch_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading dataset ...")
    datasets.config.HF_DATASETS_DOWNLOAD_TIMEOUT = 3000
    fineweb = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",split="train", streaming=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_id, torch_dtype=torch.float16, device_map="auto").eval()
    chat_model = AutoModelForCausalLM.from_pretrained(config.chat_model_id, torch_dtype=torch.float16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    extract_diffs_streaming(fineweb,
                            base_model,
                            chat_model,
                            tokenizer,
                            target_tokens=100_000_000,
                            batch_size=32,
                            layer_idx=config.target_layer, 
                            save_frequency=5_000_000
                            )

    sae = DiffSAE(input_dim=config.hidden_dim, dict_size=config.hidden_dim * 8, k=64, auxk=256).to(device)
    train_diffsae_chunked(
        sae,
        chunk_dir="diff_activations",      
        num_chunks=20,            
        num_epochs=20,                    
        batch_size=4096,
        lr=1e-4,
        aux_coef=1/32,
        log_interval=100,
        save_dir="diffsae_checkpoints",
        device=device
    )


if __name__ == "__main__":
    main()  