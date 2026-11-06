"""
Train & Benchmark a Small Next-Concept Model (NCM) vs. Autoregressive LLM
=========================================================================
Runs end-to-end training and evaluation on a structured 64-sentence semantic corpus
with a constrained codebook (V_k = 8 codes per position) so that:
- Level 1 (l_1 = 1, max 8 states) MUST learn macro-level topic concepts ("blurry outline")
- Level 2 (l_2 = 2) refines subject/action clause structure
- Level 3 (l_3 = 4) refines phrase-level choices
- Level 4 (l_4 = 8) refines modifier words
- Level 5 (l_5 = 16) resolves 100% of exact surface tokens!
"""

import json
import time
from typing import List, Dict, Tuple

import torch

from ncm_model import (
    NCMConfig,
    MultiscaleResidualVQVAE,
    NextConceptTransformer,
    AutoregressiveTransformer,
)


def build_concept_corpus() -> Tuple[List[List[str]], Dict[str, int], Dict[int, str]]:
    """
    Build 64 distinct 16-word sentences across 4 macro domains x 4 mechanisms x 4 outcomes.
    Because there are 64 sentences and Level-1 has only l_1=1 code with codebook_size=8,
    Level 1 cannot memorize individual sentences—it is forced to compress global concepts
    first, while finer levels (l_2=2, l_3=4, l_4=8, l_5=16) progressively resolve details!
    """
    domains = [
        # (Domain prefix [4 words])
        ["next", "concept", "language", "models"],
        ["parallel", "hierarchical", "neural", "networks"],
        ["modern", "gpu", "tensor", "clusters"],
        ["autonomous", "symbolic", "reasoning", "agents"],
    ]
    mechanisms = [
        # (Mechanism clause [6 words])
        ["compress", "complex", "ideas", "into", "discrete", "codes"],
        ["sketch", "coarse", "semantic", "outlines", "in", "parallel"],
        ["refine", "multiscale", "latent", "representations", "across", "levels"],
        ["bypass", "slow", "sequential", "token", "memory", "bottlenecks"],
    ]
    outcomes = [
        # (Outcome clause [6 words]) -> 4 + 6 + 6 = 16 words total
        ["to", "generate", "accurate", "responses", "much", "faster"],
        ["while", "preserving", "global", "logical", "sentence", "structure"],
        ["for", "efficient", "real", "time", "inference", "tasks"],
        ["and", "solve", "hard", "structured", "planning", "problems"],
    ]

    tokenized_sents: List[List[str]] = []
    vocab_set = {"<pad>", "<unk>"}

    for d_idx, dom in enumerate(domains):
        for m_idx, mech in enumerate(mechanisms):
            for o_idx, out in enumerate(outcomes):
                words = dom + mech + out
                assert len(words) == 16
                tokenized_sents.append(words)
                vocab_set.update(words)

    sorted_vocab = ["<pad>", "<unk>"] + sorted(vocab_set - {"<pad>", "<unk>"})
    word2idx = {w: i for i, w in enumerate(sorted_vocab)}
    idx2word = {i: w for i, w in enumerate(sorted_vocab)}
    return tokenized_sents, word2idx, idx2word


def decode_tokens(token_ids: torch.Tensor, idx2word: Dict[int, str]) -> str:
    return " ".join(idx2word.get(int(t), "<unk>") for t in token_ids.tolist())


def highlight_diffs(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, idx2word: Dict[int, str]) -> str:
    """Format decoded tokens, marking words that don't yet match target in [brackets]."""
    out = []
    for p, t in zip(pred_tokens.tolist(), target_tokens.tolist()):
        w = idx2word.get(int(p), "<unk>")
        if p == t:
            out.append(w)
        else:
            out.append(f"[{w}]")
    return " ".join(out)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 84)
    print("  NEXT-CONCEPT MODELING (NCM) — COARSE-TO-FINE EXPLORATION & PROTOTYPE BENCHMARK")
    print("=" * 84)

    sents, word2idx, idx2word = build_concept_corpus()
    seq_len = 16
    vocab_size = len(word2idx)

    # Convert corpus to tensor (64, 16)
    data = torch.tensor([[word2idx[w] for w in s] for s in sents], dtype=torch.long, device=device)
    print(f"[Corpus] {len(sents)} combinatorial 16-word sentences | Vocab size: {vocab_size} words")
    print(f"[Hierarchy] Scales l = (1, 2, 4, 8, 16) with Codebook Size V_k = 8 per scale")

    cfg = NCMConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        scales=(1, 2, 4, 8, 16),
        codebook_size=8,  # Constrained codebook (8 codes) forces true coarse-to-fine hierarchy!
        d_model=64,
        n_heads=4,
        n_vq_layers=2,
        n_gen_layers=3,
        commitment_beta=0.25,
    )

    # -------------------------------------------------------------------------
    # STAGE 1: Train Multiscale Residual VQ-VAE (Hierarchical Concept Tokenizer)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 84)
    print("STAGE 1: Training Multiscale Residual VQ-VAE (Scales l = 1, 2, 4, 8, 16 | V_k = 8)")
    print("-" * 84)
    vqvae = MultiscaleResidualVQVAE(cfg).to(device)
    vq_opt = torch.optim.AdamW(vqvae.parameters(), lr=3e-3, weight_decay=1e-4)

    for step in range(1, 501):
        vq_opt.zero_grad()
        out = vqvae(data)
        out["loss"].backward()
        vq_opt.step()

        if step % 125 == 0 or step == 1:
            preds = out["logits"].argmax(dim=-1)
            acc = (preds == data).float().mean().item() * 100.0
            print(
                f"  Step {step:3d}/500 | Total Loss: {out['loss'].item():.4f} "
                f"| Recon Loss: {out['recon_loss'].item():.4f} "
                f"| VQ Loss: {out['vq_loss'].item():.4f} | Final Scale Token Acc: {acc:5.1f}%"
            )

    # Evaluate Coarse-to-Fine Reconstruction Accuracy at each scale k = 1..5
    vqvae.eval()
    with torch.no_grad():
        z = vqvae.encode_continuous(data)
        all_indices, cumulative_z_hats, _ = vqvae.quantize_multiscale(z)

        scale_recon_accuracies = {}
        print("\n[Coarse-to-Fine Reconstruction Accuracy Across Concept Scales (64 Sentences)]")
        for k, l_k in enumerate(cfg.scales):
            logits_k = vqvae.decode_from_latent(cumulative_z_hats[k])
            preds_k = logits_k.argmax(dim=-1)
            acc_k = (preds_k == data).float().mean().item() * 100.0
            unique_codes = len(torch.unique(all_indices[k]))
            scale_recon_accuracies[f"scale_{k+1}_l{l_k}"] = round(acc_k, 2)
            print(
                f"  Level k={k+1} (l_{k+1}={l_k:2d} codes | {unique_codes}/{cfg.codebook_size} active codebook entries) "
                f"-> Token Reconstruction Acc: {acc_k:5.1f}%"
            )

        # Show a concrete example of Coarse-to-Fine decoding for Sentence #11
        sample_idx = 11
        print(f"\n[Visualizing Coarse-to-Fine Concept Decoding on Sentence #{sample_idx}]")
        print("  (Words in [brackets] are blurry coarse guesses before fine residual codes arrive)")
        print(f"  Ground Truth : {decode_tokens(data[sample_idx], idx2word)}")
        coarse_to_fine_example = {}
        for k, l_k in enumerate(cfg.scales):
            _, logits_k = vqvae.reconstruct_from_codes(
                [c[sample_idx : sample_idx + 1] for c in all_indices], up_to_scale=k + 1
            )
            pred_toks = logits_k[0].argmax(dim=-1)
            decoded_str = decode_tokens(pred_toks, idx2word)
            diff_str = highlight_diffs(pred_toks, data[sample_idx], idx2word)
            match_cnt = int((pred_toks == data[sample_idx]).sum().item())
            codes_list = all_indices[k][sample_idx].tolist()
            coarse_to_fine_example[f"level_{k+1}_l{l_k}"] = {
                "codes": codes_list,
                "matched_tokens": f"{match_cnt}/16",
                "decoded_text": decoded_str,
                "diff_view": diff_str,
            }
            print(f"  Level {k+1} (l={l_k:2d}, {match_cnt:2d}/16 exact) -> {diff_str}")

    # -------------------------------------------------------------------------
    # STAGE 2: Train Next-Concept Parallel Generator vs. Autoregressive Baseline
    # -------------------------------------------------------------------------
    print("\n" + "-" * 84)
    print("STAGE 2: Training Parallel Next-Concept Transformer (NCM) vs. Autoregressive LLM")
    print("-" * 84)

    # Use 6-token prompt (Domain + first 2 words of Mechanism) or 5-token prompt
    prompt_len = 6
    ncm_gen = NextConceptTransformer(cfg, num_prompt_tokens=prompt_len).to(device)
    ncm_opt = torch.optim.AdamW(ncm_gen.parameters(), lr=3e-3, weight_decay=1e-4)

    for step in range(1, 451):
        ncm_opt.zero_grad()
        loss, scale_accs = ncm_gen.compute_training_loss(vqvae, data)
        loss.backward()
        ncm_opt.step()

        if step % 150 == 0 or step == 1:
            mean_code_acc = sum(scale_accs.values()) / len(scale_accs) * 100.0
            print(
                f"  [NCM Generator] Step {step:3d}/450 | Concept Cross-Entropy: {loss.item():.4f} "
                f"| Parallel Code Acc: {mean_code_acc:5.1f}%"
            )

    # Train Matched Autoregressive Next-Token Baseline
    ar_model = AutoregressiveTransformer(cfg).to(device)
    ar_opt = torch.optim.AdamW(ar_model.parameters(), lr=3e-3, weight_decay=1e-4)

    for step in range(1, 451):
        ar_opt.zero_grad()
        ar_loss = ar_model.compute_loss(data)
        ar_loss.backward()
        ar_opt.step()

    # -------------------------------------------------------------------------
    # STAGE 3: Parallel Coarse-to-Fine Generation Demo
    # -------------------------------------------------------------------------
    print("\n" + "-" * 84)
    print("STAGE 3: Parallel Level-by-Level Generation Demo (5 Parallel Passes Total)")
    print("-" * 84)

    ncm_gen.eval()
    ar_model.eval()

    test_indices = [0, 21, 42, 63]
    prompts = data[test_indices, :prompt_len]
    gen_out = ncm_gen.generate_coarse_to_fine(vqvae, prompts, temperature=0.0)

    generation_examples = []
    for idx_pos, corpus_idx in enumerate(test_indices):
        prompt_str = decode_tokens(prompts[idx_pos], idx2word)
        target_str = decode_tokens(data[corpus_idx], idx2word)
        ncm_final_str = decode_tokens(gen_out["final_tokens"][idx_pos], idx2word)
        print(f"\n  Sample #{corpus_idx:2d} | Prompt: '{prompt_str} ...'")
        step_traces = []
        for k, l_k in enumerate(cfg.scales):
            step_toks = gen_out["intermediate_tokens"][k][idx_pos]
            step_diff = highlight_diffs(step_toks, data[corpus_idx], idx2word)
            match_cnt = int((step_toks == data[corpus_idx]).sum().item())
            step_traces.append(decode_tokens(step_toks, idx2word))
            print(f"    Pass {k+1}/5 (l_{k+1}={l_k:2d} parallel codes | {match_cnt:2d}/16 match): {step_diff}")
        generation_examples.append({
            "corpus_idx": corpus_idx,
            "prompt": prompt_str,
            "target": target_str,
            "ncm_passes": step_traces,
            "ncm_generated": ncm_final_str,
        })

    # -------------------------------------------------------------------------
    # STAGE 4: Concept Code Steering (Swapping Only Level-1 Coarse Code r_1)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 84)
    print("STAGE 4: Concept Code Steering (Swapping Level-1 Global Concept Code r_1)")
    print("-" * 84)
    with torch.no_grad():
        base_idx = 0   # Domain 0: "next concept language models ..."
        donor_idx = 48 # Domain 3: "autonomous symbolic reasoning agents ..."
        base_codes = [c[base_idx : base_idx + 1].clone() for c in all_indices]
        donor_l1_code = all_indices[0][donor_idx : donor_idx + 1].clone()
        swapped_codes = [donor_l1_code] + [c[base_idx : base_idx + 1].clone() for c in all_indices[1:]]

        _, orig_l1_logits = vqvae.reconstruct_from_codes(base_codes, up_to_scale=1)
        _, swap_l1_logits = vqvae.reconstruct_from_codes(swapped_codes, up_to_scale=1)
        _, orig_l3_logits = vqvae.reconstruct_from_codes(base_codes, up_to_scale=3)
        _, swap_l3_logits = vqvae.reconstruct_from_codes(swapped_codes, up_to_scale=3)

        print(f"  Original Level-1 Sketch (r_1={base_codes[0].item()}): {decode_tokens(orig_l1_logits[0].argmax(dim=-1), idx2word)}")
        print(f"  Swapped  Level-1 Sketch (r_1={donor_l1_code.item()}): {decode_tokens(swap_l1_logits[0].argmax(dim=-1), idx2word)}")
        print(f"  Original Level-3 Refinement       : {decode_tokens(orig_l3_logits[0].argmax(dim=-1), idx2word)}")
        print(f"  Swapped  Level-3 Refinement       : {decode_tokens(swap_l3_logits[0].argmax(dim=-1), idx2word)}")

    # -------------------------------------------------------------------------
    # STAGE 5: Inference Speed & Forward Pass (NFE) Scaling Benchmark
    # -------------------------------------------------------------------------
    print("\n" + "-" * 84)
    print("STAGE 5: Inference Speed & Forward Pass (NFE) Scaling Benchmark")
    print("-" * 84)

    scaling_results = []
    for test_L, test_scales in [
        (16, (1, 2, 4, 8, 16)),
        (64, (1, 2, 4, 8, 16, 32, 64)),
        (128, (1, 2, 4, 8, 16, 32, 64, 128)),
        (256, (1, 2, 4, 8, 16, 32, 64, 128, 256)),
    ]:
        test_cfg = NCMConfig(
            vocab_size=vocab_size,
            seq_len=test_L,
            scales=test_scales,
            codebook_size=8,
            d_model=64,
            n_heads=4,
            n_vq_layers=2,
            n_gen_layers=3,
        )
        test_vq = MultiscaleResidualVQVAE(test_cfg).to(device).eval()
        test_ncm = NextConceptTransformer(test_cfg, num_prompt_tokens=3).to(device).eval()
        test_ar = AutoregressiveTransformer(test_cfg).to(device).eval()
        dummy_prompt = torch.zeros((4, 3), dtype=torch.long, device=device)

        # Warmup
        _ = test_ncm.generate_coarse_to_fine(test_vq, dummy_prompt)
        _ = test_ar.generate_autoregressive(dummy_prompt)

        n_runs = 10
        t_ncm_start = time.perf_counter()
        for _ in range(n_runs):
            ncm_res = test_ncm.generate_coarse_to_fine(test_vq, dummy_prompt)
        ncm_ms = (time.perf_counter() - t_ncm_start) * 1000.0 / n_runs

        t_ar_start = time.perf_counter()
        for _ in range(n_runs):
            _, ar_passes = test_ar.generate_autoregressive(dummy_prompt)
        ar_ms = (time.perf_counter() - t_ar_start) * 1000.0 / n_runs

        ncm_passes = ncm_res["num_forward_passes"]
        speedup = ar_ms / ncm_ms
        pass_reduction = ar_passes / ncm_passes

        print(
            f"  SeqLen L={test_L:3d} | AR Passes: {ar_passes:3d} ({ar_ms:6.2f} ms) "
            f"| NCM Parallel Passes: {ncm_passes:2d} ({ncm_ms:6.2f} ms) "
            f"| Speedup: {speedup:5.2f}x ({pass_reduction:4.1f}x fewer passes)"
        )
        scaling_results.append({
            "seq_len": test_L,
            "scales": list(test_scales),
            "ar_forward_passes": ar_passes,
            "ncm_forward_passes": ncm_passes,
            "ar_latency_ms": round(ar_ms, 2),
            "ncm_latency_ms": round(ncm_ms, 2),
            "speedup_x": round(speedup, 2),
            "pass_reduction_x": round(pass_reduction, 2),
        })

    summary = {
        "scale_reconstruction_accuracy_pct": scale_recon_accuracies,
        "coarse_to_fine_example": coarse_to_fine_example,
        "generation_examples": generation_examples,
        "scaling_benchmark": scaling_results,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved experiment metrics and coarse-to-fine traces to /data/workspace/exploration/results.json")


if __name__ == "__main__":
    main()
