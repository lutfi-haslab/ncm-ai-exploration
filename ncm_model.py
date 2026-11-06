"""
Next-Concept Modeling (NCM) — Small PyTorch Implementation
==========================================================
Inspired by "Next-Concept Modeling" (Kumbong et al., Stanford Hazy Research & Together AI, 2026)
arXiv:2610.14162

Core Two-Stage Architecture:
1. MultiscaleResidualVQVAE (The Hierarchical Concept Compressor):
   Maps a token sequence x of length L into K hierarchical discrete code levels
   with progressively finer sequence lengths: l_1 < l_2 < ... < l_K = L.
   - Coarse scales (e.g., l_1 = 1, l_2 = 2) capture global sentence concepts / outlines.
   - Fine scales (e.g., l_K = L) capture exact lexical tokens and syntax residual details.
   - Uses cumulative reconstruction: z_hat^(k) = sum_{j=1}^k R_j(e_j(r_j))
   - Uses progressive scale supervision so every intermediate scale k produces a valid
     "blurry-to-sharp" text reconstruction when decoded.

2. NextConceptTransformer (The Coarse-to-Fine Parallel Generator):
   Instead of generating L tokens sequentially one-by-one (like standard Autoregressive LLMs),
   NCM predicts an entire scale level r_k of length l_k IN PARALLEL in a single forward pass,
   conditioned on the prompt and cumulative reconstruction z_hat^(k-1) from coarser levels.
   Total sequential forward passes = K (where K << L).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NCMConfig:
    vocab_size: int = 128
    seq_len: int = 16
    scales: Tuple[int, ...] = (1, 2, 4, 8, 16)  # l_1 < l_2 < ... < l_K = L
    codebook_size: int = 64                     # V_k discrete concept codes per scale
    d_model: int = 64
    n_heads: int = 4
    n_vq_layers: int = 2
    n_gen_layers: int = 3
    commitment_beta: float = 0.25


class VectorQuantizer(nn.Module):
    """
    Discrete codebook for a single scale level k with Straight-Through Estimator (STE)
    and automatic data-driven initialization on first batch to prevent codebook collapse.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _init_from_data(self, flat_z: torch.Tensor) -> None:
        """Initialize codebook entries from the first batch of continuous residual vectors."""
        n = flat_z.shape[0]
        if n >= self.num_embeddings:
            perm = torch.randperm(n, device=flat_z.device)[: self.num_embeddings]
            self.embedding.weight.data.copy_(flat_z[perm])
        else:
            repeats = (self.num_embeddings + n - 1) // n
            tiled = flat_z.repeat(repeats, 1)[: self.num_embeddings]
            noise = torch.randn_like(tiled) * 0.05
            self.embedding.weight.data.copy_(tiled + noise)
        self.initialized.fill_(True)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: Continuous residual tensor of shape (B, l_k, d_model)
        Returns:
            z_q_ste: Quantized vectors with straight-through gradients (B, l_k, d_model)
            indices: Discrete code indices r_k of shape (B, l_k)
            vq_loss: Codebook + commitment loss scalar
        """
        B, l_k, D = z_e.shape
        flat_z = z_e.reshape(-1, D)  # (B * l_k, D)

        if self.training and not bool(self.initialized.item()):
            self._init_from_data(flat_z.detach())

        # Compute L2 distance between continuous latents and codebook vectors
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        indices = torch.argmin(distances, dim=1)  # (B * l_k,)
        z_q = self.embedding(indices).view(B, l_k, D)
        indices = indices.view(B, l_k)

        # VQ objective: codebook loss + beta * commitment loss
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-Through Estimator (pass gradients directly from z_q back to z_e)
        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, indices, vq_loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Map discrete code indices (B, l_k) -> vectors (B, l_k, D)."""
        return self.embedding(indices)


class ScaleResizer(nn.Module):
    """
    Downsampling D_k (L -> l_k) and Upsampling/Reconstruction operator R_k (l_k -> L)
    for mapping between full sequence resolution L and scale level resolution l_k.
    Preserves coordinate alignment with residual r^(k) = z - z_hat^(k-1) via residual refinement.
    """

    def __init__(self, seq_len: int, scale_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.scale_len = scale_len
        # Residual convolutional refinement R_k(u) = u + conv(u) initialized near identity
        self.up_refine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        nn.init.zeros_(self.up_refine[-1].weight)
        nn.init.zeros_(self.up_refine[-1].bias)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample residual from (B, L, D) -> (B, l_k, D) via average pooling."""
        if self.scale_len == self.seq_len:
            return x
        x_t = x.transpose(1, 2)  # (B, D, L)
        return F.adaptive_avg_pool1d(x_t, self.scale_len).transpose(1, 2)

    def upsample(self, z_q: torch.Tensor) -> torch.Tensor:
        """Reconstruction operator R_k: Upsample (B, l_k, D) -> (B, L, D)."""
        z_t = z_q.transpose(1, 2)  # (B, D, l_k)
        if self.scale_len != self.seq_len:
            if self.scale_len == 1:
                z_t = z_t.expand(-1, -1, self.seq_len)
            else:
                z_t = F.interpolate(z_t, size=self.seq_len, mode="nearest")
        refined = z_t + self.up_refine(z_t)
        return refined.transpose(1, 2)  # (B, L, D)


class MultiscaleResidualVQVAE(nn.Module):
    """
    Stage 1 of NCM: Multiscale Residual VQ-VAE Tokenizer.
    Compresses a sequence of tokens x in {0..V-1}^L into K discrete code levels
    r_1, ..., r_K with lengths l_1 < l_2 < ... < l_K = L.
    """

    def __init__(self, cfg: NCMConfig):
        super().__init__()
        self.cfg = cfg
        self.num_scales = len(cfg.scales)

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_vq_layers, enable_nested_tensor=False
        )
        self.enc_norm = nn.LayerNorm(cfg.d_model)

        # One VectorQuantizer and ScaleResizer per scale level k = 1..K
        self.quantizers = nn.ModuleList([
            VectorQuantizer(cfg.codebook_size, cfg.d_model, beta=cfg.commitment_beta)
            for _ in cfg.scales
        ])
        self.resizers = nn.ModuleList([
            ScaleResizer(cfg.seq_len, l_k, cfg.d_model)
            for l_k in cfg.scales
        ])
        # Positional bias added during decoding so even l_1=1 can decode into structured positions
        self.dec_pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            dec_layer, num_layers=cfg.n_vq_layers, enable_nested_tensor=False
        )
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size)

    def encode_continuous(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode input tokens (B, L) into continuous latent representation z in R^{B x L x D}."""
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        z = self.enc_norm(self.encoder(x))
        return z

    def quantize_multiscale(
        self, z: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Perform multiscale residual quantization across K levels:
            r^(k) = z - z_hat^(k-1)
            z_e^(k) = Downsample_k(r^(k))
            z_q^(k), indices_k = Quantize_k(z_e^(k))
            z_hat^(k) = z_hat^(k-1) + R_k(z_q^(k))

        Returns:
            all_indices: List of K tensors, each of shape (B, l_k)
            cumulative_z_hats: List of K tensors, each of shape (B, L, D) representing z_hat^(k)
            total_vq_loss: Scalar sum of VQ losses across all K scales
        """
        z_hat = torch.zeros_like(z)
        all_indices: List[torch.Tensor] = []
        cumulative_z_hats: List[torch.Tensor] = []
        total_vq_loss = torch.tensor(0.0, device=z.device)

        for k in range(self.num_scales):
            # Subtract detached cumulative reconstruction so each scale quantizes clean residual
            residual = z - z_hat.detach()
            z_e_k = self.resizers[k].downsample(residual)
            z_q_k, indices_k, vq_loss_k = self.quantizers[k](z_e_k)
            upsampled_k = self.resizers[k].upsample(z_q_k)

            z_hat = z_hat + upsampled_k
            all_indices.append(indices_k)
            cumulative_z_hats.append(z_hat)
            total_vq_loss = total_vq_loss + vq_loss_k

        return all_indices, cumulative_z_hats, total_vq_loss

    def decode_from_latent(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Decode any cumulative latent reconstruction z_hat^(k) (B, L, D) -> token logits (B, L, V)."""
        B, L, _ = z_hat.shape
        positions = torch.arange(L, device=z_hat.device).unsqueeze(0).expand(B, L)
        h = z_hat + self.dec_pos_emb(positions)
        h = self.dec_norm(self.decoder(h))
        return self.to_logits(h)

    def reconstruct_from_codes(
        self, codes_per_scale: List[torch.Tensor], up_to_scale: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct cumulative latent z_hat^(k) and token logits from discrete concept codes
        up to scale level `up_to_scale` (1-indexed, defaults to all K scales).
        """
        max_k = up_to_scale if up_to_scale is not None else len(codes_per_scale)
        B = codes_per_scale[0].shape[0]
        device = codes_per_scale[0].device
        z_hat = torch.zeros(B, self.cfg.seq_len, self.cfg.d_model, device=device)

        for k in range(max_k):
            z_q_k = self.quantizers[k].lookup(codes_per_scale[k])
            z_hat = z_hat + self.resizers[k].upsample(z_q_k)

        logits = self.decode_from_latent(z_hat)
        return z_hat, logits

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full training step of Multiscale Residual VQ-VAE.
        Applies progressive multi-scale reconstruction loss (nested scale supervision)
        so coarse levels learn high-level semantic summaries while fine levels refine exact tokens.
        """
        z = self.encode_continuous(tokens)
        all_indices, cumulative_z_hats, vq_loss = self.quantize_multiscale(z)

        # Compute reconstruction loss at the final scale K AND auxiliary coarse scales
        # (Nested scale dropout / multi-scale supervision from NCM)
        recon_loss = torch.tensor(0.0, device=tokens.device)
        scale_weights = [0.2, 0.25, 0.35, 0.5, 1.0][: self.num_scales]

        for k, z_hat_k in enumerate(cumulative_z_hats):
            logits_k = self.decode_from_latent(z_hat_k)
            ce_k = F.cross_entropy(logits_k.reshape(-1, self.cfg.vocab_size), tokens.reshape(-1))
            recon_loss = recon_loss + scale_weights[k] * ce_k

        total_loss = recon_loss + 0.25 * vq_loss

        final_logits = self.decode_from_latent(cumulative_z_hats[-1])
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "logits": final_logits,
            "indices": all_indices,
            "cumulative_z_hats": cumulative_z_hats,
        }


class NextConceptTransformer(nn.Module):
    """
    Stage 2 of NCM: Coarse-to-Fine Parallel Next-Concept Generator.
    Models:
        P(r_k | r_{<k}, prompt) = prod_{i=1}^{l_k} P(r_{k, i} | z_hat^(k-1), prompt)
    At each level k in {1, ..., K}, predicts all l_k discrete concept codes IN PARALLEL
    in a single bidirectional Transformer forward pass!
    """

    def __init__(self, cfg: NCMConfig, num_prompt_tokens: int = 3):
        super().__init__()
        self.cfg = cfg
        self.num_scales = len(cfg.scales)
        self.num_prompt_tokens = num_prompt_tokens

        # Prompt token embeddings (for conditional generation from a prefix of tokens)
        self.prompt_token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prompt_pos_emb = nn.Embedding(num_prompt_tokens, cfg.d_model)

        # Scale embedding and full-resolution positional embedding
        self.scale_emb = nn.Embedding(self.num_scales, cfg.d_model)
        self.seq_pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        # Bidirectional Transformer over full latent canvas (B, L, D)
        gen_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            gen_layer, num_layers=cfg.n_gen_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        # Scale-specific pooling & parallel classification heads for each scale k
        self.scale_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.codebook_size)
            for _ in cfg.scales
        ])

    def predict_scale_parallel(
        self,
        prompt_tokens: torch.Tensor,
        z_hat_prev: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        """
        Single parallel forward pass for scale level `scale_idx` (k).
        Args:
            prompt_tokens: Prefix conditioning tokens of shape (B, P)
            z_hat_prev: Cumulative reconstruction z_hat^(k-1) from coarser scales (B, L, D)
                        (zeros when k == 0)
            scale_idx: Current scale index k in {0, ..., K-1}
        Returns:
            code_logits: Parallel logits for all l_k codes at scale k, shape (B, l_k, V_k)
        """
        B, L, _ = z_hat_prev.shape
        P = prompt_tokens.shape[1]
        l_k = self.cfg.scales[scale_idx]

        # Build prompt conditioning tokens
        p_pos = torch.arange(P, device=prompt_tokens.device).unsqueeze(0).expand(B, P)
        prompt_h = self.prompt_token_emb(prompt_tokens) + self.prompt_pos_emb(p_pos)

        # Build canvas input from cumulative coarser reconstruction + scale & position embeddings
        seq_pos = torch.arange(L, device=z_hat_prev.device).unsqueeze(0).expand(B, L)
        scale_vec = self.scale_emb(torch.full((B, L), scale_idx, device=z_hat_prev.device, dtype=torch.long))
        canvas_h = z_hat_prev + self.seq_pos_emb(seq_pos) + scale_vec

        # Concatenate [prompt_h, canvas_h] -> bidirectional attention across entire concept canvas
        full_h = torch.cat([prompt_h, canvas_h], dim=1)  # (B, P + L, D)
        out_h = self.norm(self.transformer(full_h))
        canvas_out = out_h[:, P:, :]  # (B, L, D)

        # Pool canvas_out to scale length l_k and predict all l_k discrete codes in parallel
        if l_k < L:
            pooled = F.adaptive_avg_pool1d(canvas_out.transpose(1, 2), l_k).transpose(1, 2)
        else:
            pooled = canvas_out

        code_logits = self.scale_heads[scale_idx](pooled)  # (B, l_k, codebook_size)
        return code_logits

    def compute_training_loss(
        self,
        vqvae: MultiscaleResidualVQVAE,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Train the Next-Concept Generator across all K scales using teacher-forced
        coarse-to-fine cumulative reconstructions from the frozen VQ-VAE.
        """
        with torch.no_grad():
            z = vqvae.encode_continuous(tokens)
            target_indices, cumulative_z_hats, _ = vqvae.quantize_multiscale(z)

        B, L = tokens.shape
        prompt_tokens = tokens[:, : self.num_prompt_tokens]

        total_loss = torch.tensor(0.0, device=tokens.device)
        scale_accs: Dict[str, float] = {}

        for k in range(self.num_scales):
            if k == 0:
                z_hat_prev = torch.zeros(B, L, self.cfg.d_model, device=tokens.device)
            else:
                z_hat_prev = cumulative_z_hats[k - 1].detach()

            logits_k = self.predict_scale_parallel(prompt_tokens, z_hat_prev, k)
            targets_k = target_indices[k]  # (B, l_k)

            loss_k = F.cross_entropy(logits_k.reshape(-1, self.cfg.codebook_size), targets_k.reshape(-1))
            total_loss = total_loss + loss_k

            preds_k = logits_k.argmax(dim=-1)
            acc_k = (preds_k == targets_k).float().mean().item()
            scale_accs[f"scale_{k+1}_l{self.cfg.scales[k]}"] = acc_k

        return total_loss / self.num_scales, scale_accs

    @torch.no_grad()
    def generate_coarse_to_fine(
        self,
        vqvae: MultiscaleResidualVQVAE,
        prompt_tokens: torch.Tensor,
        temperature: float = 0.0,
    ) -> Dict[str, object]:
        """
        Inference loop of NCM:
        Executes exactly K parallel forward passes (k = 1 .. K).
        At each step k:
          1. Predicts all l_k discrete concept codes in parallel: r_k ~ P(r_k | z_hat^(k-1), prompt)
          2. Updates cumulative reconstruction: z_hat^(k) = z_hat^(k-1) + R_k(e_k(r_k))
          3. Decodes z_hat^(k) to inspect the intermediate coarse-to-fine text!
        """
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        z_hat = torch.zeros(B, self.cfg.seq_len, self.cfg.d_model, device=device)

        predicted_codes: List[torch.Tensor] = []
        intermediate_tokens: List[torch.Tensor] = []

        for k in range(self.num_scales):
            logits_k = self.predict_scale_parallel(prompt_tokens, z_hat, k)
            if temperature <= 0.0:
                codes_k = logits_k.argmax(dim=-1)  # Greedy parallel decoding (B, l_k)
            else:
                probs = F.softmax(logits_k / temperature, dim=-1)
                codes_k = torch.multinomial(probs.view(-1, self.cfg.codebook_size), 1).view(B, -1)

            predicted_codes.append(codes_k)

            # Update cumulative concept reconstruction z_hat^(k)
            z_q_k = vqvae.quantizers[k].lookup(codes_k)
            z_hat = z_hat + vqvae.resizers[k].upsample(z_q_k)

            # Decode intermediate scale reconstruction to observe coarse-to-fine progression
            step_logits = vqvae.decode_from_latent(z_hat)
            step_tokens = step_logits.argmax(dim=-1)
            intermediate_tokens.append(step_tokens)

        return {
            "final_tokens": intermediate_tokens[-1],
            "intermediate_tokens": intermediate_tokens,
            "predicted_codes": predicted_codes,
            "num_forward_passes": self.num_scales,
        }


class AutoregressiveTransformer(nn.Module):
    """
    Standard Next-Token Autoregressive LLM Baseline (GPT-style causal Transformer)
    with matched hidden dimension and depth for direct comparison with NCM.
    Requires L - P sequential forward passes to generate a sequence of length L.
    """

    def __init__(self, cfg: NCMConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.n_gen_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(tokens) + self.pos_emb(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)
        h = self.norm(self.transformer(h, mask=causal_mask, is_causal=True))
        return self.head(h)

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens[:, :-1])
        targets = tokens[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def generate_autoregressive(self, prompt_tokens: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Generate token-by-token sequentially up to cfg.seq_len."""
        seq = prompt_tokens.clone()
        num_passes = 0
        while seq.shape[1] < self.cfg.seq_len:
            logits = self.forward(seq)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            seq = torch.cat([seq, next_token], dim=1)
            num_passes += 1
        return seq, num_passes
