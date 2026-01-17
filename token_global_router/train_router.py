"""
Training Script for Token + Global Context MoE Router

This script trains ONLY the routing components:
- Context encoder (GlobalContextEncoder)
- Local router
- Global router
- Fusion gate

The expert MLPs are kept FROZEN (they're already fine-tuned).

Usage:
    python train_router.py --num_epochs 3 --batch_size 4 --output_dir ./trained_router
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from datetime import datetime
import random

from token_global_moe import build_token_global_moe, EXPERT_MODELS


class MultiDomainDataset(Dataset):
    """
    Dataset combining multiple domains for router training.
    Each sample includes domain label for analysis.
    """
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
                # GSM8K for math
                ds = load_dataset("gsm8k", "main", split="train")
                for item in ds.select(range(min(num_samples, len(ds)))):
                    samples.append(f"Question: {item['question']}\nAnswer: {item['answer']}")

            elif domain == "code":
                # CodeAlpaca or similar
                try:
                    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        text = f"Instruction: {item['instruction']}\n"
                        if item.get('input'):
                            text += f"Input: {item['input']}\n"
                        text += f"Output: {item['output']}"
                        samples.append(text)
                except:
                    # Fallback to simple code prompts
                    ds = load_dataset("openai_humaneval", split="test")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        samples.append(item['prompt'] + item['canonical_solution'])

            elif domain == "medical":
                # Medical QA
                try:
                    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
                    for item in ds.select(range(min(num_samples, len(ds)))):
                        samples.append(f"Question: {item['input']}\nAnswer: {item['output']}")
                except:
                    # Fallback
                    samples = [
                        "What are the symptoms of diabetes? The main symptoms include increased thirst, frequent urination, and fatigue.",
                        "How is hypertension treated? Treatment includes lifestyle changes and medications like ACE inhibitors.",
                    ] * (num_samples // 2)

            elif domain == "instruct":
                # Alpaca or instruction data
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
                # General text
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

        # Tokenize
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
    """
    Compute load balancing loss to prevent expert collapse.
    Encourages uniform distribution of tokens across experts.
    """
    # router_logits: (batch, seq, num_experts)
    num_experts = router_logits.size(-1)

    # Get routing probabilities
    router_probs = F.softmax(router_logits, dim=-1)

    # Mask out padding
    mask = attention_mask.unsqueeze(-1).float()
    router_probs = router_probs * mask

    # Compute expert utilization (fraction of tokens assigned to each expert)
    # Taking argmax to get hard assignment
    expert_assignments = router_probs.argmax(dim=-1)  # (batch, seq)

    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    total_tokens = attention_mask.sum()

    for i in range(num_experts):
        expert_counts[i] = ((expert_assignments == i) * attention_mask).sum()

    # Target is uniform distribution
    target_count = total_tokens / num_experts

    # L2 loss on deviation from uniform
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
    output_dir: str = "./trained_router",
    gradient_accumulation_steps: int = 4,
    save_steps: int = 500,
    logging_steps: int = 50,
):
    """Train the router components."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Freeze expert MLPs, only train router components
    # Convert trainable params to float32 for gradient scaling compatibility
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if any(x in name for x in ['context_encoder', 'local_router', 'global_router', 'gate']):
            param.requires_grad = True
            param.data = param.data.float()  # Convert to float32
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

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    total_lm_loss = 0
    total_lb_loss = 0

    # Mixed precision training
    scaler = GradScaler()
    use_amp = torch.cuda.is_available()

    print(f"\nStarting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Mixed precision: {use_amp}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            # Labels for language modeling (shifted input_ids)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # Forward pass
            # We need to capture router logits for load balancing loss
            # Move inputs to model device first
            model_device = next(model.model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            labels = labels.to(model_device)

            model._compute_context(input_ids, attention_mask)

            # Inject context into MoE layers
            all_router_logits = []
            for layer_idx, moe in model.moe_layers.items():
                moe._injected_context = model._global_context

                # Store original forward to capture router logits
                original_forward = moe.forward

                def forward_with_logging(hs, m=moe, logits_list=all_router_logits):
                    batch_size, seq_len, hidden_dim = hs.shape
                    device = hs.device

                    # Move routers to correct device if needed
                    if m.local_router.weight.device != device:
                        m.local_router = m.local_router.to(device)
                        m.global_router = m.global_router.to(device)
                        if hasattr(m, 'gate') and m.gate is not None:
                            m.gate = m.gate.to(device)

                    # Move context to correct device
                    context = m._injected_context
                    if context.device != device:
                        context = context.to(device)

                    context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)

                    local_logits = m.local_router(hs)
                    global_logits = m.global_router(context_expanded)

                    if m.fusion == "gate":
                        combined = torch.cat([hs, context_expanded], dim=-1)
                        alpha = torch.sigmoid(m.gate(combined))
                        router_logits = alpha * local_logits + (1 - alpha) * global_logits
                    else:
                        router_logits = local_logits + global_logits

                    # Clamp logits for numerical stability
                    router_logits = router_logits.clamp(-50, 50)
                    logits_list.append(router_logits)

                    # Continue with normal forward
                    router_probs = F.softmax(router_logits, dim=-1)
                    top_probs, top_indices = torch.topk(router_probs, m.top_k, dim=-1)
                    top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)

                    expert_outputs = []
                    for expert in m.experts:
                        # Move expert to correct device if needed
                        expert_device = next(expert.parameters()).device
                        if expert_device != device:
                            expert = expert.to(device)
                        expert_outputs.append(expert(hs))
                    expert_outputs = torch.stack(expert_outputs, dim=2)

                    top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
                    selected_outputs = torch.gather(expert_outputs, dim=2, index=top_indices_expanded)
                    output = (selected_outputs * top_probs.unsqueeze(-1)).sum(dim=2)

                    return output

                moe.forward = forward_with_logging

            # Run model forward with mixed precision
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
                        # Move attention_mask to router_logits device for computation
                        mask_for_lb = attention_mask.to(router_logits.device)
                        lb_component = compute_load_balancing_loss(router_logits.float(), mask_for_lb)
                        # Move result to same device as lb_loss
                        lb_loss = lb_loss + lb_component.to(lb_loss.device)
                    lb_loss = lb_loss / len(all_router_logits)

                # Total loss
                loss = lm_loss.float() + load_balance_weight * lb_loss
                loss = loss / gradient_accumulation_steps

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss at step {step}, skipping batch")
                optimizer.zero_grad()
                all_router_logits.clear()
                continue

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            total_loss += loss.item() * gradient_accumulation_steps
            total_lm_loss += lm_loss.item()
            total_lb_loss += lb_loss.item()
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_steps += 1

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # Step with scaler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
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

                # Save checkpoint
                if global_step % save_steps == 0:
                    save_checkpoint(model, output_path / f"checkpoint-{global_step}")

            # Clear router logits list
            all_router_logits.clear()

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        save_checkpoint(model, output_path / f"epoch-{epoch+1}")

    # Save final model
    save_checkpoint(model, output_path / "final")
    print(f"\nTraining completed! Model saved to {output_path}")

    return model


def save_checkpoint(model, path: Path):
    """Save only the trainable router components."""
    path.mkdir(parents=True, exist_ok=True)

    # Save context encoder
    torch.save(
        model.context_encoder.state_dict(),
        path / "context_encoder.pt"
    )

    # Save MoE layer components
    moe_states = {}
    for layer_idx, moe in model.moe_layers.items():
        moe_states[layer_idx] = {
            'local_router': moe.local_router.state_dict(),
            'global_router': moe.global_router.state_dict(),
        }
        if hasattr(moe, 'gate'):
            moe_states[layer_idx]['gate'] = moe.gate.state_dict()

    torch.save(moe_states, path / "moe_routers.pt")

    # Save config
    config = {
        'moe_layer_indices': list(model.moe_layers.keys()),
        'context_dim': model.context_encoder.context_dim,
        'num_experts': len(list(model.moe_layers.values())[0].experts),
    }
    with open(path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved checkpoint to {path}")


def load_checkpoint(model, path: Path):
    """Load trained router components."""
    # Load context encoder
    context_state = torch.load(path / "context_encoder.pt", map_location=model.device)
    model.context_encoder.load_state_dict(context_state)

    # Load MoE components
    moe_states = torch.load(path / "moe_routers.pt", map_location=model.device)
    for layer_idx, state in moe_states.items():
        if layer_idx in model.moe_layers:
            moe = model.moe_layers[layer_idx]
            moe.local_router.load_state_dict(state['local_router'])
            moe.global_router.load_state_dict(state['global_router'])
            if 'gate' in state and hasattr(moe, 'gate'):
                moe.gate.load_state_dict(state['gate'])

    print(f"Loaded checkpoint from {path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Token+Global MoE Router")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_samples_per_domain", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./trained_router")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--load_balance_weight", type=float, default=0.01)
    parser.add_argument("--context_method", type=str, default="momentum",
                       choices=["momentum", "thor"])
    parser.add_argument("--fusion_method", type=str, default="gate",
                       choices=["gate", "add"])
    parser.add_argument("--experts", type=str, nargs="+", default=None)

    args = parser.parse_args()

    print("="*60)
    print("Token + Global MoE Router Training")
    print("="*60)

    # Select experts
    if args.experts:
        expert_models = {k: v for k, v in EXPERT_MODELS.items() if k in args.experts}
    else:
        expert_models = EXPERT_MODELS

    print(f"Experts: {list(expert_models.keys())}")

    # Build model
    model, tokenizer = build_token_global_moe(
        expert_models=expert_models,
        context_method=args.context_method,
        fusion_method=args.fusion_method,
    )

    # Create dataset
    dataset = MultiDomainDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_samples_per_domain=args.num_samples_per_domain,
        domains=["math", "code", "instruct", "general"],  # Skip medical if loading fails
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
