"""
Training Script for Embedding Momentum MoE

Trains ONLY the token-level router weights.
- Embedding router: FROZEN (pre-trained embedding model + expert centroids)
- Token router: TRAINABLE
- Expert MLPs: FROZEN

Usage:
    python train_embedding_momentum.py --num_epochs 3 --batch_size 4 --output_dir ./trained_momentum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from datetime import datetime
import random

from embedding_momentum_moe import (
    build_embedding_momentum_moe,
    EXPERT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
)


class MultiDomainDataset(Dataset):
    """Dataset combining multiple domains for router training."""
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        num_samples_per_domain: int = 1000,
        domains: list = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if domains is None:
            domains = ["math", "code", "medical", "instruct", "general"]

        print("Loading multi-domain dataset...")

        for domain in domains:
            domain_samples = self._load_domain_data(domain, num_samples_per_domain)
            for text in domain_samples:
                self.samples.append({"text": text, "domain": domain})

        random.shuffle(self.samples)
        print(f"Total samples: {len(self.samples)}")

    def _load_domain_data(self, domain: str, num_samples: int) -> list:
        """Load data for a specific domain."""
        samples = []

        try:
            if domain == "math":
                ds = load_dataset("gsm8k", "main", split="train")
                for item in ds.select(range(min(num_samples, len(ds)))):
                    samples.append(f"Question: {item['question']}\nAnswer: {item['answer']}")

            elif domain == "code":
                try:
                    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        text = f"Instruction: {item['instruction']}\n"
                        if item.get('input'):
                            text += f"Input: {item['input']}\n"
                        text += f"Output: {item['output']}"
                        samples.append(text)
                except:
                    ds = load_dataset("openai_humaneval", split="test")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        samples.append(item['prompt'] + item['canonical_solution'])

            elif domain == "medical":
                try:
                    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        samples.append(f"Question: {item['input']}\nAnswer: {item['output']}")
                except:
                    samples = [
                        "What are the symptoms of diabetes? The main symptoms include increased thirst, frequent urination, and fatigue.",
                    ] * min(100, num_samples)

            elif domain == "instruct":
                try:
                    ds = load_dataset("tatsu-lab/alpaca", split="train")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        text = f"Instruction: {item['instruction']}\n"
                        if item.get('input'):
                            text += f"Input: {item['input']}\n"
                        text += f"Response: {item['output']}"
                        samples.append(text)
                except:
                    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        samples.append(f"Instruction: {item['instruction']}\nResponse: {item['response']}")

            elif domain == "general":
                try:
                    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                    texts = [item['text'] for item in ds if len(item['text']) > 100]
                    samples = texts[:num_samples]
                except:
                    samples = ["This is general text for training."] * num_samples

            print(f"  {domain}: {len(samples)} samples")

        except Exception as e:
            print(f"  {domain}: Error loading - {e}, using fallback")
            samples = [f"This is a {domain} example."] * min(100, num_samples)

        return samples[:num_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        domain = sample["domain"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "domain": domain,
        }


def compute_load_balancing_loss(router_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute load balancing loss to prevent expert collapse."""
    num_experts = router_logits.size(-1)
    router_probs = F.softmax(router_logits, dim=-1)

    mask = attention_mask.unsqueeze(-1).float()
    router_probs = router_probs * mask

    expert_assignments = router_probs.argmax(dim=-1)
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    total_tokens = attention_mask.sum()

    for i in range(num_experts):
        expert_counts[i] = ((expert_assignments == i) * attention_mask).sum()

    target_count = total_tokens / num_experts
    load_balance_loss = ((expert_counts - target_count) ** 2).mean() / (target_count ** 2 + 1e-9)

    return load_balance_loss


def train_router(
    model,
    tokenizer,
    train_dataset,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,
    load_balance_weight: float = 0.01,
    output_dir: str = "./trained_momentum",
    gradient_accumulation_steps: int = 4,
    save_steps: int = 500,
    logging_steps: int = 50,
):
    """Train the token router components."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Freeze everything except token routers
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        # Only train token_router weights
        if 'token_router' in name:
            param.requires_grad = True
            param.data = param.data.float()  # Convert to float32 for stability
            trainable_params.append(param)
        else:
            param.requires_grad = False
            frozen_params.append(param)

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in frozen_params):,}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Scheduler
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training setup
    model.train()
    scaler = GradScaler()
    use_amp = torch.cuda.is_available()

    print(f"\nStarting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Mixed precision: {use_amp}")

    # Get model device
    if torch.cuda.is_available():
        model_device = torch.device("cuda:0")
    else:
        model_device = next(model.model.parameters()).device
    print(f"  Model device: {model_device}")

    # Pre-move token routers to correct device
    for moe in model.moe_layers.values():
        moe.token_router = moe.token_router.to(model_device)

    global_step = 0
    total_loss = 0
    total_lm_loss = 0
    total_lb_loss = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # Compute momentum bias (frozen, no grad)
            with torch.no_grad():
                model.compute_momentum(input_ids, attention_mask)

            # Capture router logits for load balancing
            all_router_logits = []

            for layer_idx, moe in model.moe_layers.items():
                original_forward = moe.forward

                def forward_with_logging(hs, m=moe, logits_list=all_router_logits):
                    batch_size, seq_len, hidden_dim = hs.shape

                    # Token-level routing
                    router_dtype = m.token_router.weight.dtype
                    hs_for_routing = hs.to(router_dtype)
                    token_logits = m.token_router(hs_for_routing)

                    # Add momentum bias
                    if m._momentum_bias is not None:
                        momentum = m._momentum_bias.to(hs.device).to(router_dtype)
                        momentum_expanded = momentum.unsqueeze(1).expand(-1, seq_len, -1)
                        momentum_logits = torch.log(momentum_expanded + 1e-9)
                        router_logits = token_logits + m.momentum_weight * momentum_logits
                    else:
                        router_logits = token_logits

                    router_logits = router_logits.clamp(-50, 50)
                    logits_list.append(router_logits)

                    # Softmax and top-k
                    router_probs = F.softmax(router_logits, dim=-1)
                    top_probs, top_indices = torch.topk(router_probs, m.top_k, dim=-1)
                    top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)

                    # Expert computation
                    expert_outputs = torch.stack([expert(hs) for expert in m.experts], dim=2)

                    top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
                    selected_outputs = torch.gather(expert_outputs, dim=2, index=top_indices_expanded)
                    top_probs_matched = top_probs.to(selected_outputs.dtype)
                    output = (selected_outputs * top_probs_matched.unsqueeze(-1)).sum(dim=2)

                    return output

                moe.forward = forward_with_logging

            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                lm_loss = outputs.loss

                # Load balancing loss
                lb_loss = torch.tensor(0.0, device=lm_loss.device, dtype=torch.float32)
                if all_router_logits and load_balance_weight > 0:
                    for router_logits in all_router_logits:
                        mask_for_lb = attention_mask.to(router_logits.device)
                        lb_component = compute_load_balancing_loss(router_logits.float(), mask_for_lb)
                        lb_loss = lb_loss + lb_component.to(lb_loss.device)
                    lb_loss = lb_loss / len(all_router_logits)

                loss = lm_loss.float() + load_balance_weight * lb_loss
                loss = loss / gradient_accumulation_steps

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss at step {step}, skipping batch")
                optimizer.zero_grad()
                all_router_logits.clear()
                continue

            # Backward
            scaler.scale(loss).backward()

            total_loss += loss.item() * gradient_accumulation_steps
            total_lm_loss += lm_loss.item()
            total_lb_loss += lb_loss.item()
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_steps += 1

            # Gradient accumulation step
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    avg_lm = total_lm_loss / (logging_steps * gradient_accumulation_steps)
                    avg_lb = total_lb_loss / (logging_steps * gradient_accumulation_steps)

                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lm': f'{avg_lm:.4f}',
                        'lb': f'{avg_lb:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                    total_loss = 0
                    total_lm_loss = 0
                    total_lb_loss = 0

                if global_step % save_steps == 0:
                    save_checkpoint(model, output_path / f"checkpoint-{global_step}")

            all_router_logits.clear()

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        save_checkpoint(model, output_path / f"epoch-{epoch+1}")

    # Save final
    save_checkpoint(model, output_path / "final")
    print(f"\nTraining completed! Model saved to {output_path}")

    return model


def save_checkpoint(model, path: Path):
    """Save token router weights."""
    path.mkdir(parents=True, exist_ok=True)

    # Save token routers
    router_states = {}
    for layer_idx, moe in model.moe_layers.items():
        router_states[layer_idx] = moe.token_router.state_dict()

    torch.save(router_states, path / "token_routers.pt")

    # Save config
    config = {
        'moe_layer_indices': list(model.moe_layers.keys()),
        'momentum_weight': list(model.moe_layers.values())[0].momentum_weight,
        'num_experts': list(model.moe_layers.values())[0].num_experts,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved checkpoint to {path}")


def load_checkpoint(model, path: Path):
    """Load token router weights."""
    router_states = torch.load(path / "token_routers.pt", map_location=model.device)

    for layer_idx, state in router_states.items():
        if layer_idx in model.moe_layers:
            model.moe_layers[layer_idx].token_router.load_state_dict(state)

    print(f"Loaded checkpoint from {path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Embedding Momentum MoE Router")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_samples_per_domain", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./trained_momentum")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--load_balance_weight", type=float, default=0.01)
    parser.add_argument("--momentum_weight", type=float, default=0.5)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--experts", type=str, nargs="+", default=None)

    args = parser.parse_args()

    print("="*60)
    print("Embedding Momentum MoE Router Training")
    print("="*60)

    # Select experts
    if args.experts:
        expert_models = {k: v for k, v in EXPERT_MODELS.items() if k in args.experts}
    else:
        expert_models = EXPERT_MODELS

    print(f"Experts: {list(expert_models.keys())}")

    # Build model
    model, tokenizer = build_embedding_momentum_moe(
        expert_models=expert_models,
        embedding_model=args.embedding_model,
        momentum_weight=args.momentum_weight,
    )

    # Create dataset
    dataset = MultiDomainDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_samples_per_domain=args.num_samples_per_domain,
        domains=["math", "code", "instruct", "general"],
    )

    # Train
    model = train_router(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_balance_weight=args.load_balance_weight,
        output_dir=args.output_dir,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
