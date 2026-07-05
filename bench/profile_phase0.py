"""Phase 0: Profile flashspec's verification step to confirm the target op.

Usage
-----
    python bench/profile_phase0.py

Outputs
-------
- Console: ranked timing breakdown of each sub-operation
- bench/results/phase0_<datestamp>.txt: saved report for the ADR

What we're looking for
----------------------
The acceptance criterion kernel runs on EVERY draft token, every step.
The residual distribution path (_sample_residual) runs only on rejection —
its amortized cost is lower. We profile both under realistic parameters to
confirm which dominates before committing to a kernel selection.

Realistic parameters (from flashspec README / typical speculative decoding):
  batch_size: 1-8
  gamma (speculation depth): 4-8
  vocab_size: 32000 (LLaMA tokenizer)
"""

from __future__ import annotations

import time
import statistics
import datetime
import pathlib
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _make_inputs(batch_size: int, gamma: int, vocab_size: int, device: str = "cpu"):
    """Generate realistic float32 log-prob tensors."""
    rng = torch.Generator()
    rng.manual_seed(42)

    # Simulate realistic log-prob distributions (peaked, not uniform)
    raw_q = torch.randn(batch_size, gamma, vocab_size, generator=rng)
    raw_p = raw_q + 0.1 * torch.randn(batch_size, gamma, vocab_size, generator=rng)
    lp_q = torch.log_softmax(raw_q, dim=-1)
    lp_p = torch.log_softmax(raw_p, dim=-1)

    # Draft token ids: argmax of draft (greedy draft)
    draft_token_ids = lp_q.argmax(dim=-1)  # (B, gamma)

    u = torch.rand(batch_size, gamma, generator=rng)

    return lp_q, lp_p, draft_token_ids, u


def _timeit(fn, n_warmup: int = 10, n_iters: int = 500) -> tuple[float, float]:
    """Return (mean_us, std_us) for fn() over n_iters calls."""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000.0)  # ns → µs

    return statistics.mean(times), statistics.stdev(times)


# ---------------------------------------------------------------------------
# The individual sub-operations we want to rank
# ---------------------------------------------------------------------------

def op_gather_logprobs(lp_q, lp_p, draft_token_ids):
    """Gather scalar log-probs at the draft token positions."""
    ids = draft_token_ids.unsqueeze(-1)          # (B, γ, 1)
    lp_q_tok = lp_q.gather(-1, ids).squeeze(-1)  # (B, γ)
    lp_p_tok = lp_p.gather(-1, ids).squeeze(-1)
    return lp_q_tok, lp_p_tok


def op_acceptance_criterion(lp_q_tok, lp_p_tok, u):
    """Core hot path: log_ratio → exp → clamp → compare.

    This is: accepted = u < min(1, exp(log_p - log_q))
    Numerically equivalent to u < p/q but avoids underflow.
    """
    log_ratio = lp_p_tok - lp_q_tok        # subtraction
    accept_prob = log_ratio.exp().clamp(max=1.0)  # exp + clamp
    accepted = u < accept_prob             # comparison → bool mask
    return accepted


def op_first_rejection(accepted):
    """Argmin scan to find first rejection per sequence."""
    rejected = ~accepted
    has_rejection = rejected.any(dim=-1)
    first_rej_raw = rejected.int().argmax(dim=-1)
    return torch.where(
        has_rejection,
        first_rej_raw,
        torch.full_like(first_rej_raw, accepted.shape[1]),
    ).to(torch.int32)


def op_residual_distribution(lp_p, lp_q, first_rejection, vocab_size):
    """Adjusted distribution: max(0, p-q) / sum(max(0, p-q)).

    Only runs at rejection positions — amortized cost lower than acceptance.
    """
    batch_size, gamma, _ = lp_p.shape
    p = lp_p.exp()
    q = lp_q.exp()
    safe_pos = first_rejection.long().clamp(max=gamma - 1)
    idx = safe_pos[:, None, None].expand(batch_size, 1, vocab_size)
    p_at_rej = p.gather(1, idx).squeeze(1)   # (B, V)
    q_at_rej = q.gather(1, idx).squeeze(1)
    residual = torch.clamp(p_at_rej - q_at_rej, min=0.0)
    denom = residual.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    residual = residual / denom
    return residual


def op_full_pipeline(lp_q, lp_p, draft_token_ids, u):
    """Complete verification pipeline for end-to-end reference."""
    lp_q_tok, lp_p_tok = op_gather_logprobs(lp_q, lp_p, draft_token_ids)
    accepted = op_acceptance_criterion(lp_q_tok, lp_p_tok, u)
    first_rej = op_first_rejection(accepted)
    residual = op_residual_distribution(lp_p, lp_q, first_rej, lp_p.shape[-1])
    return accepted, first_rej, residual


# ---------------------------------------------------------------------------
# Main profiling run
# ---------------------------------------------------------------------------

def profile_scenario(
    label: str,
    batch_size: int,
    gamma: int,
    vocab_size: int,
    n_iters: int = 500,
) -> dict:
    lp_q, lp_p, draft_token_ids, u = _make_inputs(batch_size, gamma, vocab_size)

    # Pre-compute intermediate values so each sub-op is timed in isolation
    lp_q_tok, lp_p_tok = op_gather_logprobs(lp_q, lp_p, draft_token_ids)
    accepted = op_acceptance_criterion(lp_q_tok, lp_p_tok, u)
    first_rej = op_first_rejection(accepted)

    results = {}

    ops = [
        ("gather_logprobs",
         lambda: op_gather_logprobs(lp_q, lp_p, draft_token_ids)),
        ("acceptance_criterion",
         lambda: op_acceptance_criterion(lp_q_tok, lp_p_tok, u)),
        ("first_rejection",
         lambda: op_first_rejection(accepted)),
        ("residual_distribution",
         lambda: op_residual_distribution(lp_p, lp_q, first_rej, vocab_size)),
        ("full_pipeline",
         lambda: op_full_pipeline(lp_q, lp_p, draft_token_ids, u)),
    ]

    print(f"\n{'='*60}")
    print(f"Scenario: {label}  (B={batch_size}, γ={gamma}, V={vocab_size})")
    print(f"{'='*60}")
    print(f"{'Operation':<30} {'Mean (µs)':>10} {'Std (µs)':>10} {'% of pipeline':>14}")
    print(f"{'-'*65}")

    # Get pipeline baseline first
    pipe_mean, _ = _timeit(lambda: op_full_pipeline(lp_q, lp_p, draft_token_ids, u),
                           n_iters=n_iters)

    for name, fn in ops[:-1]:  # skip full_pipeline in the loop
        mean_us, std_us = _timeit(fn, n_iters=n_iters)
        pct = 100.0 * mean_us / pipe_mean
        results[name] = {"mean_us": mean_us, "std_us": std_us, "pct_of_pipeline": pct}
        print(f"  {name:<28} {mean_us:>10.2f} {std_us:>10.2f} {pct:>13.1f}%")

    print(f"  {'full_pipeline':<28} {pipe_mean:>10.2f} {'':>10} {'100.0':>13}%")
    results["full_pipeline"] = {"mean_us": pipe_mean}

    return results


def main():
    # Typical speculative decoding scenarios
    scenarios = [
        ("single-seq-short-gamma",  1,  4, 32000),
        ("single-seq-long-gamma",   1,  8, 32000),
        ("batch8-short-gamma",      8,  4, 32000),
        ("batch8-long-gamma",       8,  8, 32000),
        ("batch8-llama3-vocab",     8,  4, 128256),  # LLaMA-3 tokenizer
    ]

    all_results = {}
    for label, B, gamma, V in scenarios:
        all_results[label] = profile_scenario(label, B, gamma, V)

    # Summary: which op dominates across scenarios?
    print(f"\n{'='*60}")
    print("SUMMARY — mean % of pipeline time per op across scenarios")
    print(f"{'='*60}")
    op_names = ["gather_logprobs", "acceptance_criterion", "first_rejection", "residual_distribution"]
    for op in op_names:
        pcts = [all_results[s][op]["pct_of_pipeline"] for s in all_results]
        print(f"  {op:<30} avg={statistics.mean(pcts):5.1f}%  min={min(pcts):5.1f}%  max={max(pcts):5.1f}%")

    # Write dated report
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"phase0_{stamp}.txt"

    lines = [
        "FlashSpec-ASM Phase 0 Profiling Report",
        f"Generated: {datetime.datetime.now().isoformat()}",
        f"Platform: CPU (torch {torch.__version__}, numpy {np.__version__})",
        "",
        "NOTE: This is CPU-path profiling. The Triton kernel runs on GPU;",
        "CPU profiling identifies the op's arithmetic structure, not its GPU cost.",
        "The target op for AVX2 assembly is the CPU-fallback path of the same logic.",
        "",
    ]
    for label, results in all_results.items():
        lines.append(f"Scenario: {label}")
        for op, vals in results.items():
            if op == "full_pipeline":
                lines.append(f"  full_pipeline: {vals['mean_us']:.2f} µs")
            else:
                lines.append(
                    f"  {op}: {vals['mean_us']:.2f} µs ± {vals['std_us']:.2f}  "
                    f"({vals['pct_of_pipeline']:.1f}% of pipeline)"
                )
        lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")
    print("\nPhase 0 conclusion: see report for kernel selection rationale.")


if __name__ == "__main__":
    main()
