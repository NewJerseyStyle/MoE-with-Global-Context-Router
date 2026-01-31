"""
Embedding-based Momentum MoE

Uses a dedicated embedding model to compute semantic similarity between
input and expert profiles, then biases token-level routing accordingly.

Key components:
1. Embedding model (e.g., Qwen3-Embedding-0.6B) for semantic vectors
2. Pre-computed expert embeddings (centroids from representative samples)
3. Cosine similarity -> softmax -> routing bias
4. Token-level router + global bias = final routing

This is the "momentum" approach: global semantic context biases local routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import gc


# Expert model paths
EXPERT_MODELS = {
    "base": "Qwen/Qwen3-0.6B",
    "medical": "suayptalha/Qwen3-0.6B-Medical-Expert",
    "code": "suayptalha/Qwen3-0.6B-Code-Expert",
    "math": "suayptalha/Qwen3-0.6B-Math-Expert",
    "instruct": "suayptalha/Qwen3-0.6B-IF-Expert",
}
# Default embedding model
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class ExpertEmbeddingRouter(nn.Module):
    """
    Routes based on cosine similarity between input embedding and expert embeddings.
    Returns a bias to add to token-level router logits.
    """
    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        num_experts: int = 5,
        expert_names: List[str] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_names = expert_names or [f"expert_{i}" for i in range(num_experts)]
        self.temperature = temperature

        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_name,
            trust_remote_code=True
        )
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.embedding_model.eval()

        # Expert embeddings will be set later
        self.register_buffer('expert_embeddings', None)
        self.embedding_dim = None

    def set_expert_embeddings(self, embeddings: torch.Tensor, expert_names: List[str] = None):
        """
        Set pre-computed expert embeddings.

        Args:
            embeddings: (num_experts, embedding_dim) normalized embeddings
            expert_names: list of expert names in same order
        """
        self.expert_embeddings = F.normalize(embeddings, dim=-1)
        self.embedding_dim = embeddings.shape[-1]
        self.num_experts = embeddings.shape[0]
        if expert_names:
            self.expert_names = expert_names
        print(f"Set {self.num_experts} expert embeddings with dim {self.embedding_dim}")

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings using the embedding model."""
        device = next(self.embedding_model.parameters()).device

        inputs = self.embedding_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        outputs = self.embedding_model(**inputs)

        # Mean pooling over non-padding tokens
        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def compute_expert_embeddings(
        self,
        domain_samples: Dict[str, List[str]],
        samples_per_domain: int = 50
    ) -> torch.Tensor:
        """
        Compute expert embeddings from representative samples.

        Args:
            domain_samples: Dict mapping expert name to list of sample texts
            samples_per_domain: Max samples to use per domain

        Returns:
            Expert embeddings tensor (num_experts, embedding_dim)
        """
        expert_embeddings = []

        for expert_name in self.expert_names:
            if expert_name not in domain_samples:
                print(f"  Warning: No samples for {expert_name}, using zero embedding")
                if self.embedding_dim:
                    expert_embeddings.append(torch.zeros(self.embedding_dim))
                else:
                    # Get embedding dim from a dummy encode
                    dummy = self.encode(["test"])
                    self.embedding_dim = dummy.shape[-1]
                    expert_embeddings.append(torch.zeros(self.embedding_dim))
                continue

            samples = domain_samples[expert_name][:samples_per_domain]
            print(f"  Encoding {len(samples)} samples for {expert_name}...")

            # Encode in batches
            all_embeddings = []
            batch_size = 8
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                emb = self.encode(batch)
                all_embeddings.append(emb.cpu())

            # Average to get centroid
            centroid = torch.cat(all_embeddings, dim=0).mean(dim=0)
            expert_embeddings.append(centroid)

        embeddings = torch.stack(expert_embeddings, dim=0)
        self.set_expert_embeddings(embeddings.to(next(self.embedding_model.parameters()).device))
        return embeddings

    def forward(self, input_text: str = None, input_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        Compute routing bias based on input similarity to experts.

        Args:
            input_text: Text to encode (if input_embedding not provided)
            input_embedding: Pre-computed embedding (batch, embedding_dim)

        Returns:
            Routing bias (batch, num_experts) - softmax of cosine similarities
        """
        if self.expert_embeddings is None:
            raise ValueError("Expert embeddings not set. Call compute_expert_embeddings first.")

        if input_embedding is None:
            if input_text is None:
                raise ValueError("Either input_text or input_embedding must be provided")
            input_embedding = self.encode([input_text] if isinstance(input_text, str) else input_text)

        # Move expert embeddings to same device
        expert_emb = self.expert_embeddings.to(input_embedding.device)

        # Cosine similarity
        similarities = torch.matmul(input_embedding, expert_emb.T)  # (batch, num_experts)

        # Apply temperature and softmax to get routing bias
        routing_bias = F.softmax(similarities / self.temperature, dim=-1)

        return routing_bias


class MoELayerWithMomentum(nn.Module):
    """
    MoE layer with embedding-based momentum routing.

    Combines:
    - Token-level router: local routing based on hidden states
    - Global momentum: bias from semantic similarity to expert profiles
    """
    def __init__(
        self,
        experts: nn.ModuleList,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        momentum_weight: float = 0.5,
    ):
        super().__init__()
        self.experts = experts
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.momentum_weight = momentum_weight

        # Token-level router
        self.token_router = nn.Linear(hidden_dim, num_experts)

        # Momentum bias will be injected from outside
        self._momentum_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq, hidden_dim)

        Returns:
            output: (batch, seq, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Token-level routing
        router_dtype = self.token_router.weight.dtype
        hs_for_routing = hidden_states.to(router_dtype)
        token_logits = self.token_router(hs_for_routing)  # (batch, seq, num_experts)

        # Add momentum bias if available
        if self._momentum_bias is not None:
            # momentum_bias is (batch, num_experts), expand to (batch, seq, num_experts)
            momentum = self._momentum_bias.to(device).to(router_dtype)
            momentum_expanded = momentum.unsqueeze(1).expand(-1, seq_len, -1)

            # Add bias (momentum as log-probability adjustment)
            # Using log to convert softmax output to logit-space adjustment
            momentum_logits = torch.log(momentum_expanded + 1e-9)
            router_logits = token_logits + self.momentum_weight * momentum_logits
        else:
            router_logits = token_logits

        # Clamp for stability
        router_logits = router_logits.clamp(-50, 50)

        # Softmax and top-k
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Expert computation
        expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=2)

        # Gather selected experts
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        selected_outputs = torch.gather(expert_outputs, dim=2, index=top_indices_expanded)

        # Weighted combination
        top_probs_matched = top_probs.to(selected_outputs.dtype)
        output = (selected_outputs * top_probs_matched.unsqueeze(-1)).sum(dim=2)

        return output


class EmbeddingMomentumMoE(nn.Module):
    """
    Full MoE model with embedding-based momentum routing.
    """
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        expert_mlps: Dict[int, nn.ModuleList],
        embedding_router: ExpertEmbeddingRouter,
        momentum_weight: float = 0.5,
        top_k: int = 2,
    ):
        super().__init__()
        self.model = base_model
        self.embedding_router = embedding_router
        self.hidden_dim = base_model.config.hidden_size

        # Create MoE layers
        self.moe_layers = nn.ModuleDict()
        self.moe_layer_indices = set()

        for layer_idx, experts in expert_mlps.items():
            moe = MoELayerWithMomentum(
                experts=experts,
                hidden_dim=self.hidden_dim,
                num_experts=len(experts),
                top_k=top_k,
                momentum_weight=momentum_weight,
            )
            self.moe_layers[str(layer_idx)] = moe
            self.moe_layer_indices.add(layer_idx)

            # Replace MLP in base model
            self.model.model.layers[layer_idx].mlp = moe

        self._momentum_bias = None

    def compute_momentum(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Compute momentum bias from input using embedding model."""
        # Decode input_ids to text
        # Note: This is a simplification - in practice you might want to batch this
        texts = self.embedding_router.embedding_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        # Get routing bias from embedding model
        self._momentum_bias = self.embedding_router(input_text=texts)

        # Inject into MoE layers
        for moe in self.moe_layers.values():
            moe._momentum_bias = self._momentum_bias

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs
    ):
        # Compute momentum bias
        self.compute_momentum(input_ids, attention_mask)

        # Forward through model
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate with momentum routing."""
        self.compute_momentum(input_ids, attention_mask)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

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


def create_domain_samples() -> Dict[str, List[str]]:
    """Create representative samples for each expert domain."""
    return {
        "base": [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Can you tell me about the weather?",
            "I would like to learn more about history.",
            "What are some good books to read?",
            "Tell me an interesting fact.",
            "How do I make a cup of coffee?",
            "What time is it in Tokyo?",
        ],
        "medical": [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "Explain the mechanism of action of aspirin.",
            "What is the difference between Type 1 and Type 2 diabetes?",
            "Describe the symptoms of pneumonia.",
            "What are common side effects of chemotherapy?",
            "How does the immune system fight infections?",
            "What is the treatment for acute myocardial infarction?",
        ],
        "code": [
            "Write a Python function to sort a list.",
            "How do I implement a binary search tree?",
            "Explain the difference between async and sync programming.",
            "What is the time complexity of quicksort?",
            "How do I connect to a PostgreSQL database in Python?",
            "Write a recursive function to calculate factorial.",
            "Explain the concept of dependency injection.",
            "What is the difference between REST and GraphQL?",
        ],
        "math": [
            "Solve the equation: 2x + 5 = 13",
            "What is the derivative of x^2 + 3x?",
            "Calculate the integral of sin(x).",
            "Prove that the square root of 2 is irrational.",
            "What is the formula for the area of a circle?",
            "Explain the Pythagorean theorem.",
            "Solve: If a train travels at 60 mph, how far in 2.5 hours?",
            "What is the sum of the first 100 natural numbers?",
        ],
        "instruct": [
            "Summarize the main points of this article.",
            "Write a professional email to request a meeting.",
            "Create a step-by-step guide for baking a cake.",
            "Help me write a cover letter for a software engineering job.",
            "Explain this concept in simple terms for a beginner.",
            "Give me feedback on my writing.",
            "How should I structure my presentation?",
            "What are the best practices for time management?",
        ],
    }


def build_embedding_momentum_moe(
    expert_models: Dict[str, str] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    moe_layer_indices: List[int] = None,
    momentum_weight: float = 0.5,
    top_k: int = 2,
    device: str = "cuda",
) -> Tuple[EmbeddingMomentumMoE, AutoTokenizer]:
    """
    Build Embedding Momentum MoE from expert models.
    """
    if expert_models is None:
        expert_models = EXPERT_MODELS

    expert_names = list(expert_models.keys())
    print(f"Building Embedding Momentum MoE with {len(expert_names)} experts: {expert_names}")

    # Load base model
    base_path = expert_models.get("base", list(expert_models.values())[0])
    print(f"\nLoading base model: {base_path}")

    if torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
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

    # Get target device
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
    else:
        target_device = next(base_model.parameters()).device
    print(f"Target device: {target_device}")

    # Convert to ModuleLists and move to device
    expert_mlps_modules = {}
    for idx, mlps in expert_mlps.items():
        module_list = nn.ModuleList(mlps)
        module_list = module_list.to(device=target_device, dtype=torch.float16)
        expert_mlps_modules[idx] = module_list

    # Create embedding router
    print("\nCreating embedding router...")
    embedding_router = ExpertEmbeddingRouter(
        embedding_model_name=embedding_model,
        num_experts=len(expert_names),
        expert_names=expert_names,
    )
    embedding_router.embedding_model = embedding_router.embedding_model.to(target_device)

    # Compute expert embeddings
    print("\nComputing expert embeddings...")
    domain_samples = create_domain_samples()
    embedding_router.compute_expert_embeddings(domain_samples)

    # Build MoE model
    print("\nBuilding EmbeddingMomentumMoE...")
    moe_model = EmbeddingMomentumMoE(
        base_model=base_model,
        expert_mlps=expert_mlps_modules,
        embedding_router=embedding_router,
        momentum_weight=momentum_weight,
        top_k=top_k,
    )

    # Move token routers to device and correct dtype
    for moe in moe_model.moe_layers.values():
        moe.token_router = moe.token_router.to(device=target_device, dtype=torch.float16)

    print(f"\nModel built successfully!")
    print(f"  Experts: {expert_names}")
    print(f"  MoE layers: {len(moe_layer_indices)}")
    print(f"  Momentum weight: {momentum_weight}")
    print(f"  Top-k: {top_k}")

    return moe_model, tokenizer


if __name__ == "__main__":
    # Test build
    print("Testing Embedding Momentum MoE build...")
    model, tokenizer = build_embedding_momentum_moe()

    # Test forward
    text = "What is the derivative of x squared?"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output shape: {outputs.logits.shape}")
    print("Test passed!")
