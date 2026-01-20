"""
Model-Level Router (MoDEM-style)

Routes entire inputs to specific expert models based on domain classification.
Uses embedding similarity or learned classifier to select experts.

Two modes:
1. Single selection: Route to best matching expert
2. Weighted ensemble: Weighted average of top-k expert outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


# Expert model definitions
EXPERT_MODELS = {
    "base": "Qwen/Qwen3-1.7B",
    "medical": "prithivMLmods/Sculptor-Qwen3_Med-Reasoning",
    "instruct": "gustavecortal/Qwen3-psychological-reasoning-1.7B",
}


class DomainClassifier(nn.Module):
    """
    Lightweight classifier to predict domain from input embedding.
    Uses mean pooling of token embeddings + MLP.
    """
    def __init__(self, hidden_dim: int, num_domains: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
        Returns:
            logits: (batch, num_domains)
        """
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        return self.classifier(pooled)


class EmbeddingSimilarityRouter:
    """
    Routes based on cosine similarity between input embedding and domain embeddings.
    No training required - uses pre-computed domain centroids.
    """
    def __init__(self, hidden_dim: int, domain_names: List[str]):
        self.hidden_dim = hidden_dim
        self.domain_names = domain_names
        self.domain_embeddings = None  # (num_domains, hidden_dim)

    def set_domain_embeddings(self, embeddings: torch.Tensor):
        """Set pre-computed domain centroid embeddings."""
        self.domain_embeddings = F.normalize(embeddings, dim=-1)

    def compute_domain_embedding(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute mean-pooled embedding for input."""
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        return F.normalize(pooled, dim=-1)

    def route(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to top-k domains based on similarity.

        Returns:
            indices: (batch, top_k) - selected domain indices
            weights: (batch, top_k) - normalized similarity weights
        """
        if self.domain_embeddings is None:
            raise ValueError("Domain embeddings not set. Call set_domain_embeddings first.")

        input_emb = self.compute_domain_embedding(hidden_states, attention_mask)

        # Move domain embeddings to same device as input
        domain_emb = self.domain_embeddings.to(input_emb.device)

        # Cosine similarity
        similarities = torch.matmul(input_emb, domain_emb.T)  # (batch, num_domains)

        # Top-k selection
        weights, indices = torch.topk(similarities, k=min(top_k, len(self.domain_names)), dim=-1)
        weights = F.softmax(weights, dim=-1)

        return indices, weights


class ModelLevelMoE:
    """
    Model-Level Mixture of Experts.

    Routes inputs to specialized expert models based on domain.
    Supports single selection or weighted ensemble.
    """
    def __init__(
        self,
        expert_models: Dict[str, str] = None,
        router_type: str = "similarity",  # "similarity" or "classifier"
        device: str = "cuda",
        load_in_4bit: bool = True,
    ):
        self.expert_models = expert_models or EXPERT_MODELS
        self.domain_names = list(self.expert_models.keys())
        self.router_type = router_type
        self.device = device
        self.load_in_4bit = load_in_4bit

        self.models: Dict[str, AutoModelForCausalLM] = {}
        self.tokenizer = None
        self.router = None
        self.hidden_dim = None

    def load_base_for_routing(self):
        """Load base model for computing embeddings (used for routing)."""
        print("Loading base model for routing...")

        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Disable flash attention for T4
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        base_path = self.expert_models.get("base", "Qwen/Qwen3-1.7B")
        self.models["base"] = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get hidden dim
        self.hidden_dim = self.models["base"].config.hidden_size
        print(f"Hidden dim: {self.hidden_dim}")

        # Initialize router
        if self.router_type == "similarity":
            self.router = EmbeddingSimilarityRouter(self.hidden_dim, self.domain_names)
        else:
            self.router = DomainClassifier(self.hidden_dim, len(self.domain_names))
            self.router = self.router.to(self.device)

    def load_expert(self, domain: str):
        """Load a specific expert model."""
        if domain in self.models:
            return

        print(f"Loading expert: {domain}...")

        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Disable flash attention for T4
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        model_path = self.expert_models[domain]
        self.models[domain] = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.models[domain].eval()

    def unload_expert(self, domain: str):
        """Unload an expert to free memory."""
        if domain in self.models and domain != "base":
            del self.models[domain]
            torch.cuda.empty_cache()

    def compute_input_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get hidden states from base model for routing."""
        # Get the device of the model's embedding layer
        model_device = next(self.models["base"].parameters()).device

        # Move inputs to the model's device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        with torch.no_grad():
            outputs = self.models["base"](
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Use first layer hidden states for efficiency
            hidden_states = outputs.hidden_states[1]  # After first layer
        return hidden_states

    def route(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, top_k: int = 1) -> Tuple[List[str], torch.Tensor]:
        """
        Determine which expert(s) to use for this input.

        Returns:
            domains: List of selected domain names
            weights: Tensor of weights for each domain
        """
        hidden_states = self.compute_input_embedding(input_ids, attention_mask)

        # Move attention_mask to same device as hidden_states
        if attention_mask is not None and attention_mask.device != hidden_states.device:
            attention_mask = attention_mask.to(hidden_states.device)

        if self.router_type == "similarity":
            indices, weights = self.router.route(hidden_states, attention_mask, top_k=top_k)
        else:
            logits = self.router(hidden_states, attention_mask)
            weights, indices = torch.topk(F.softmax(logits, dim=-1), k=top_k, dim=-1)

        # Convert indices to domain names (take first batch item)
        selected_domains = [self.domain_names[idx] for idx in indices[0].tolist()]

        return selected_domains, weights[0]

    def generate_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        domain: str,
        **generate_kwargs
    ) -> torch.Tensor:
        """Generate using a single expert."""
        self.load_expert(domain)

        # Get the device of the model
        model_device = next(self.models[domain].parameters()).device

        # Move inputs to the model's device
        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        # Set pad_token_id if not provided
        if 'pad_token_id' not in generate_kwargs:
            generate_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.models[domain].generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs
            )
        return outputs

    def generate(
        self,
        prompt: str,
        top_k: int = 1,
        ensemble: bool = False,
        **generate_kwargs
    ) -> Tuple[str, Dict]:
        """
        Generate response using routed expert(s).

        Args:
            prompt: Input text
            top_k: Number of experts to consider
            ensemble: If True, ensemble top-k outputs; if False, use top-1
            **generate_kwargs: Arguments for model.generate()

        Returns:
            generated_text: The generated response
            routing_info: Dictionary with routing decisions
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Route
        selected_domains, weights = self.route(input_ids, attention_mask, top_k=top_k)

        routing_info = {
            "selected_domains": selected_domains,
            "weights": weights.tolist(),
        }

        if ensemble and top_k > 1:
            # TODO: Implement proper ensemble (average logits)
            # For now, just use top-1
            pass

        # Generate with top expert
        top_domain = selected_domains[0]
        outputs = self.generate_single(
            input_ids, attention_mask, top_domain, **generate_kwargs
        )

        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text, routing_info

    def compute_domain_centroids(self, domain_samples: Dict[str, List[str]], save_path: Optional[str] = None):
        """
        Compute centroid embeddings for each domain from sample texts.
        Used for similarity-based routing.

        Args:
            domain_samples: Dict mapping domain name to list of sample texts
            save_path: Optional path to save centroids
        """
        if self.router_type != "similarity":
            raise ValueError("Centroids only used for similarity routing")

        print("Computing domain centroids...")
        centroids = []

        for domain in self.domain_names:
            if domain not in domain_samples:
                print(f"  Warning: No samples for {domain}, using random embedding")
                centroids.append(torch.randn(self.hidden_dim))
                continue

            samples = domain_samples[domain]
            embeddings = []

            for text in samples[:50]:  # Limit samples
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                # Don't move to device here - compute_input_embedding will handle it

                hidden = self.compute_input_embedding(inputs["input_ids"], inputs["attention_mask"])

                # Move attention_mask to same device as hidden
                mask = inputs["attention_mask"].to(hidden.device).unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                embeddings.append(pooled.cpu())

            centroid = torch.cat(embeddings, dim=0).mean(dim=0)
            centroids.append(centroid)
            print(f"  {domain}: computed from {len(embeddings)} samples")

        centroids = torch.stack(centroids, dim=0).to(self.device)
        self.router.set_domain_embeddings(centroids)

        if save_path:
            torch.save(centroids, save_path)
            print(f"Saved centroids to {save_path}")

    def load_domain_centroids(self, path: str):
        """Load pre-computed domain centroids."""
        centroids = torch.load(path, map_location=self.device)
        self.router.set_domain_embeddings(centroids)
        print(f"Loaded centroids from {path}")


def create_default_domain_samples() -> Dict[str, List[str]]:
    """Create default sample texts for each domain."""
    return {
        "base": [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Tell me about yourself.",
            "Can you help me with something?",
        ],
        "medical": [
            "What are the symptoms of diabetes?",
            "How do I treat a headache?",
            "What medication should I take for fever?",
            "Explain the difference between bacteria and viruses.",
            "What is hypertension and how is it managed?",
        ],
        "instruct": [
            "Summarize this article for me.",
            "Translate this text to French.",
            "List 5 tips for productivity.",
            "Compare and contrast these two options.",
            "Give me step-by-step instructions to bake a cake.",
        ],
    }


if __name__ == "__main__":
    # Test the router
    print("Testing Model-Level Router...")

    moe = ModelLevelMoE(router_type="similarity", load_in_4bit=True)
    moe.load_base_for_routing()

    # Compute centroids from default samples
    samples = create_default_domain_samples()
    moe.compute_domain_centroids(samples, save_path="domain_centroids.pt")

    # Test routing
    test_prompts = [
        "What are the symptoms of COVID-19?",
        "Write a Python function to calculate fibonacci.",
        "Solve: 3x + 7 = 22",
        "Hello, how are you?",
    ]

    for prompt in test_prompts:
        domains, weights = moe.route(
            *[x.to(moe.device) for x in moe.tokenizer(prompt, return_tensors="pt").values()],
            top_k=3
        )
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Routed to: {domains}")
        print(f"  Weights: {[f'{w:.3f}' for w in weights.tolist()]}")
