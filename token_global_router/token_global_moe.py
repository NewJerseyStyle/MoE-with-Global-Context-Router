"""
Token-Level Router with Global Context (Momentum/THOR-style)

Simplified implementation using forward hooks instead of rebuilding the model.

This implements a MoE where:
1. Each expert is an FFN from a different fine-tuned model
2. Token-level routing is augmented with sequence-level context

Context methods:
- "momentum": Attention-pooled embedding (our approach)
- "thor": Average of preceding hidden states (THOR-MoE style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import gc


EXPERT_MODELS = {
    "base": "Qwen/Qwen3-0.6B",
    "medical": "suayptalha/Qwen3-0.6B-Medical-Expert",
    "code": "suayptalha/Qwen3-0.6B-Code-Expert",
    "math": "suayptalha/Qwen3-0.6B-Math-Expert",
    "instruct": "suayptalha/Qwen3-0.6B-IF-Expert",
}


class GlobalContextEncoder(nn.Module):
    """Encodes sequence into global context vector."""
    def __init__(self, hidden_dim: int, context_dim: int, method: str = "momentum"):
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Attention pooling for momentum
        self.attention = nn.Linear(hidden_dim, 1)
        self.projection = nn.Linear(hidden_dim, context_dim)
        self.layer_norm = nn.LayerNorm(context_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden_states: (batch, seq, hidden)
        if self.method == "momentum":
            # Attention-weighted pooling
            attn_scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq)
            if attention_mask is not None:
                # Use float16-safe large negative value (max float16 is ~65504)
                attn_scores = attn_scores.masked_fill(~attention_mask.bool(), -65000.0)
            attn_weights = F.softmax(attn_scores, dim=-1)
            # Handle case where all positions are masked (shouldn't happen but safety)
            attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / hidden_states.size(1))
            attn_weights = attn_weights.unsqueeze(-1)
            pooled = (hidden_states * attn_weights).sum(dim=1)
        else:  # thor - simple mean
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)

        context = self.projection(pooled)
        return self.layer_norm(context)


class ContextAwareMoELayer(nn.Module):
    """
    MoE layer that replaces the original MLP.
    Uses global context for routing decisions.
    """
    def __init__(
        self,
        experts: nn.ModuleList,
        hidden_dim: int,
        context_dim: int,
        fusion: str = "gate",
        top_k: int = 2,
    ):
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.top_k = min(top_k, self.num_experts)
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # Routers
        self.local_router = nn.Linear(hidden_dim, self.num_experts)
        self.global_router = nn.Linear(context_dim, self.num_experts)

        # Fusion gate
        if fusion == "gate":
            self.gate = nn.Linear(hidden_dim + context_dim, 1)
        self.fusion = fusion

    def forward(self, hidden_states: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq, hidden_dim)
            global_context: (batch, context_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Move routers to the same device as hidden_states if needed
        if self.local_router.weight.device != device:
            self.local_router = self.local_router.to(device)
            self.global_router = self.global_router.to(device)
            if hasattr(self, 'gate') and self.gate is not None:
                self.gate = self.gate.to(device)

        # Move context to the same device
        if global_context.device != device:
            global_context = global_context.to(device)

        # Expand context to all positions
        context_expanded = global_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute router logits
        local_logits = self.local_router(hidden_states)  # (batch, seq, num_experts)
        global_logits = self.global_router(context_expanded)

        if self.fusion == "gate":
            combined = torch.cat([hidden_states, context_expanded], dim=-1)
            alpha = torch.sigmoid(self.gate(combined))
            router_logits = alpha * local_logits + (1 - alpha) * global_logits
        else:  # add
            router_logits = local_logits + global_logits

        # Clamp logits for numerical stability
        router_logits = router_logits.clamp(-50, 50)

        # Top-k routing
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)  # Renormalize

        # Compute expert outputs - move experts to correct device if needed
        expert_outputs = []
        for expert in self.experts:
            # Check if expert is on correct device
            expert_device = next(expert.parameters()).device
            if expert_device != device:
                expert = expert.to(device)
            expert_outputs.append(expert(hidden_states))
        expert_outputs = torch.stack(expert_outputs, dim=2)  # (batch, seq, num_experts, hidden)

        # Gather and weight
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        selected_outputs = torch.gather(expert_outputs, dim=2, index=top_indices_expanded)
        output = (selected_outputs * top_probs.unsqueeze(-1)).sum(dim=2)

        return output


class TokenGlobalMoE(nn.Module):
    """
    Wrapper that adds global context to a base model's MoE routing.
    Uses hooks to inject context without modifying forward pass.
    """
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        expert_mlps: Dict[int, nn.ModuleList],  # layer_idx -> list of MLP modules
        context_method: str = "momentum",
        fusion_method: str = "gate",
        context_dim: int = 256,
        top_k: int = 2,
    ):
        super().__init__()

        self.config = base_model.config
        self.hidden_dim = self.config.hidden_size

        # Keep the base model
        self.model = base_model

        # Context encoder
        self.context_encoder = GlobalContextEncoder(
            hidden_dim=self.hidden_dim,
            context_dim=context_dim,
            method=context_method,
        )

        # Replace MLPs with MoE layers
        self.moe_layers = nn.ModuleDict()
        for layer_idx, experts in expert_mlps.items():
            moe = ContextAwareMoELayer(
                experts=experts,
                hidden_dim=self.hidden_dim,
                context_dim=context_dim,
                fusion=fusion_method,
                top_k=top_k,
            )
            self.moe_layers[str(layer_idx)] = moe

            # Replace the MLP in the original model
            self.model.model.layers[layer_idx].mlp = moe

        # Storage for global context (set during forward)
        self._global_context = None

        # Register hooks to inject context
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to inject global context."""
        def make_hook(layer_idx):
            def hook(module, args, kwargs):
                # Inject global context into MoE layer
                if hasattr(module, 'forward') and self._global_context is not None:
                    # The MoE layer will receive context through closure
                    pass
                return args, kwargs
            return hook

        # We'll handle context injection differently - through the MoE layer itself

    def _compute_context(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Compute global context from input embeddings."""
        with torch.no_grad():
            embeddings = self.model.model.embed_tokens(input_ids)

        # Move context encoder to same device as embeddings if needed
        embed_device = embeddings.device
        if next(self.context_encoder.parameters()).device != embed_device:
            self.context_encoder = self.context_encoder.to(embed_device)

        self._global_context = self.context_encoder(embeddings.detach(), attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Compute global context
        self._compute_context(input_ids, attention_mask)

        # Inject context into MoE layers
        for layer_idx, moe in self.moe_layers.items():
            # Create a wrapper that includes context
            original_forward = moe.forward

            def forward_with_context(hidden_states, global_context=self._global_context, orig_fn=original_forward):
                return orig_fn(hidden_states, global_context)

            moe.forward = forward_with_context

        # Run the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # Restore original forwards (prevent memory leak)
        # Actually we recreate each time so it's fine

        self._global_context = None

        return outputs

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate with global context."""
        # Compute context once at the start
        self._compute_context(input_ids, attention_mask)

        # Inject context into MoE layers
        for layer_idx, moe in self.moe_layers.items():
            original_forward = moe.__class__.forward

            # Bind context
            moe._injected_context = self._global_context

            def forward_with_context(self, hidden_states):
                return ContextAwareMoELayer.forward(self, hidden_states, self._injected_context)

            moe.forward = lambda hs, m=moe: ContextAwareMoELayer.forward(m, hs, m._injected_context)

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


def extract_mlp_from_model(model: AutoModelForCausalLM, layer_indices: List[int]) -> Dict[int, nn.Module]:
    """Extract MLP modules from a model."""
    mlps = {}
    for idx in layer_indices:
        if idx < len(model.model.layers):
            mlps[idx] = deepcopy(model.model.layers[idx].mlp)
    return mlps


def build_token_global_moe(
    expert_models: Dict[str, str] = None,
    moe_layer_indices: List[int] = None,
    context_method: str = "momentum",
    fusion_method: str = "gate",
    context_dim: int = 256,
    top_k: int = 2,
    device: str = "cuda",
) -> Tuple[TokenGlobalMoE, AutoTokenizer]:
    """
    Build Token+Global MoE from multiple fine-tuned models.
    """
    if expert_models is None:
        expert_models = EXPERT_MODELS

    expert_names = list(expert_models.keys())
    print(f"Building MoE with {len(expert_names)} experts: {expert_names}")

    # Load base model
    base_path = expert_models.get("base", list(expert_models.values())[0])
    print(f"\nLoading base model: {base_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(base_model.model.layers)

    # Default: apply MoE to middle layers
    if moe_layer_indices is None:
        moe_layer_indices = list(range(num_layers // 4, 3 * num_layers // 4))

    print(f"MoE layers: {moe_layer_indices}")

    # Collect MLPs from each expert model
    expert_mlps = {idx: [] for idx in moe_layer_indices}

    for name, path in expert_models.items():
        print(f"Loading expert: {name} from {path}")

        if path == base_path:
            model = base_model
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation="eager",
            )

        mlps = extract_mlp_from_model(model, moe_layer_indices)

        for layer_idx, mlp in mlps.items():
            expert_mlps[layer_idx].append(mlp)

        if path != base_path:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Get target device from base model
    target_device = next(base_model.parameters()).device

    # Convert to ModuleLists and move to correct device/dtype
    expert_mlps_modules = {}
    for idx, mlps in expert_mlps.items():
        module_list = nn.ModuleList(mlps)
        # Move all expert MLPs to the correct device and dtype
        module_list = module_list.to(device=target_device, dtype=torch.float16)
        expert_mlps_modules[idx] = module_list

    # Build MoE model
    print("\nBuilding TokenGlobalMoE...")
    moe_model = TokenGlobalMoE(
        base_model=base_model,
        expert_mlps=expert_mlps_modules,
        context_method=context_method,
        fusion_method=fusion_method,
        context_dim=context_dim,
        top_k=top_k,
    )

    # Move context encoder to correct dtype
    moe_model.context_encoder = moe_model.context_encoder.to(
        device=target_device,
        dtype=torch.float16
    )

    # Move MoE routers to correct dtype
    for moe in moe_model.moe_layers.values():
        moe.local_router = moe.local_router.to(device=target_device, dtype=torch.float16)
        moe.global_router = moe.global_router.to(device=target_device, dtype=torch.float16)
        if hasattr(moe, 'gate'):
            moe.gate = moe.gate.to(device=target_device, dtype=torch.float16)

    moe_model.eval()

    print(f"Model built successfully!")
    print(f"  Context method: {context_method}")
    print(f"  Fusion method: {fusion_method}")
    print(f"  Top-k: {top_k}")
    print(f"  Num experts: {len(expert_names)}")
    print(f"  MoE layers: {len(moe_layer_indices)}")

    return moe_model, tokenizer


if __name__ == "__main__":
    print("Testing Token+Global MoE...")

    # Build model with fewer experts for testing
    test_experts = {
        "base": "Qwen/Qwen3-0.6B",
        "math": "suayptalha/Qwen3-0.6B-Math-Expert",
    }

    model, tokenizer = build_token_global_moe(
        expert_models=test_experts,
        moe_layer_indices=[10, 11, 12, 13, 14, 15],
        context_method="momentum",
        fusion_method="gate",
        top_k=2,
    )

    # Test generation
    prompt = "What is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            inputs.get("attention_mask"),
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
