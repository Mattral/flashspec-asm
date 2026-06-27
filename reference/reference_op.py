"""FlashSpec-ASM — Reference Implementation (Phase 1 ground truth).

Target operation: residual distribution computation from speculative decoding
Algorithm 1 (Leviathan et al., 2023).

Mathematical definition
-----------------------
Given:
  p[v]  — target model probability at vocab index v   (p = exp(log_p))
  q[v]  — draft  model probability at vocab index v   (q = exp(log_q))

The residual distribution is:

    residual[v] = max(0, p[v] - q[v])
    residual    = residual / sum(residual)    (normalise to a valid pmf)

This is computed at the FIRST REJECTION POSITION per sequence. For each
batch element b, the slice used is:

    p_slice[v] = p[b, first_rejection[b], v]
    q_slice[v] = q[b, first_rejection[b], v]

Edge cases (all must be handled correctly by the assembly kernel):
  - All q[v] >= p[v]     → all residuals are 0 before normalisation.
                           Numerics: denom = 0 → guard with epsilon.
  - NaN in log_p or log_q → propagates; kernel must not silently convert
                             to 0 or 1.
  - inf in log_p or log_q → exp(inf) = inf, exp(-inf) = 0; both are valid
                             float32 states that must propagate correctly.
  - Ties (p[v] == q[v])  → max(0, 0) = 0; stable, no special case needed.
  - Extreme magnitudes    → log-prob near -745 (exp underflows to 0 in f32);
                             log-prob = 0 (log-prob of certainty); both valid.

This file is the ONLY ground truth. The C baseline and ASM kernel are both
tested against this implementation — never the other way around.

ABI note
--------
The C/ASM kernel will operate on a SINGLE (vocab_size,) slice of float32
probabilities already gathered at the rejection position, i.e.:

    void residual_dist(
        const float* p,    // target probs at rejection pos, shape (V,)
        const float* q,    // draft  probs at rejection pos, shape (V,)
        float*       out,  // output residual pmf,           shape (V,)
        int64_t      V     // vocab size
    );

The Python reference implements the batched version; the single-slice
variant `residual_dist_single` matches the exact C ABI signature for
unit-testing the kernel in isolation.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "residual_dist_single",
    "residual_dist_batched",
    "gather_and_residual",
]

# Denominator guard: prevents divide-by-zero when all max(0, p-q) == 0.
# Value chosen to be safely below float32 precision (1.2e-7) but above
# subnormal range. The kernel MUST use the same guard value.
_DENOM_GUARD: float = 1e-9


# ---------------------------------------------------------------------------
# Single-slice reference (matches C ABI exactly)
# ---------------------------------------------------------------------------

def residual_dist_single(
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute residual distribution for one (p, q) pair.

    Parameters
    ----------
    p : np.ndarray
        Target model probabilities. Shape: ``(V,)``, dtype float32.
        Must be a valid probability vector (non-negative, sums to ~1).
    q : np.ndarray
        Draft model probabilities. Shape: ``(V,)``, dtype float32.
        Must be a valid probability vector (non-negative, sums to ~1).

    Returns
    -------
    np.ndarray
        Residual probability distribution. Shape: ``(V,)``, dtype float32.
        Sums to 1.0 (or 0.0 if p == q everywhere, after guard).

    Notes
    -----
    The computation in float32, step by step:

        diff   = p - q              (V,) — may be negative
        clamped = max(0, diff)      (V,) — relu
        denom  = sum(clamped)       scalar — may be 0 if p <= q everywhere
        out    = clamped / max(denom, 1e-9)

    This is the EXACT sequence the AVX2 kernel must implement:
      1. vsubps    (p - q)
      2. vmaxps    (clamp negatives to 0)
      3. horizontal sum of 8-wide ymm registers
      4. vdivps or vmulps+reciprocal    (normalise)
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError(f"p and q must be 1-D arrays of equal shape; got {p.shape}, {q.shape}")

    diff = p - q
    clamped = np.maximum(diff, 0.0, dtype=np.float32)
    denom = float(clamped.sum())
    out = clamped / max(denom, _DENOM_GUARD)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Batched reference (matches flashspec._sample_residual)
# ---------------------------------------------------------------------------

def residual_dist_batched(
    lp_p: torch.Tensor,
    lp_q: torch.Tensor,
    first_rejection: torch.Tensor,
) -> torch.Tensor:
    """Compute residual distributions for a full batch.

    Parameters
    ----------
    lp_p : torch.Tensor
        Target log-probs. Shape: ``(batch_size, gamma, vocab_size)``, float32.
    lp_q : torch.Tensor
        Draft log-probs.  Shape: ``(batch_size, gamma, vocab_size)``, float32.
    first_rejection : torch.Tensor
        First rejection index per sequence. Shape: ``(batch_size,)``, int32.
        Values in ``[0, gamma]``; value ``gamma`` means all accepted (bonus token
        position uses the target distribution at position gamma-1).

    Returns
    -------
    torch.Tensor
        Residual distributions. Shape: ``(batch_size, vocab_size)``, float32.
        Each row is a valid pmf (sums to 1.0, or to 0.0 if guard fires).
    """
    batch_size, gamma, vocab_size = lp_p.shape

    p = lp_p.float().exp()   # (B, γ, V)
    q = lp_q.float().exp()   # (B, γ, V)

    # Gather slice at first_rejection position (clamp to valid index range)
    safe_pos = first_rejection.long().clamp(max=gamma - 1)          # (B,)
    idx = safe_pos[:, None, None].expand(batch_size, 1, vocab_size)  # (B, 1, V)

    p_slice = p.gather(1, idx).squeeze(1)   # (B, V)
    q_slice = q.gather(1, idx).squeeze(1)   # (B, V)

    diff = p_slice - q_slice                         # (B, V)
    clamped = torch.clamp(diff, min=0.0)             # (B, V) relu
    denom = clamped.sum(dim=-1, keepdim=True)        # (B, 1)
    denom = denom.clamp(min=_DENOM_GUARD)            # guard
    out = clamped / denom                            # (B, V) normalised

    return out


# ---------------------------------------------------------------------------
# Full pipeline reference (gather + residual, for end-to-end tests)
# ---------------------------------------------------------------------------

def gather_and_residual(
    lp_p: torch.Tensor,
    lp_q: torch.Tensor,
    draft_token_ids: torch.Tensor,
    u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full CPU-side verification pipeline.

    This is the exact operation flashspec's CPU fallback path performs:
    acceptance test + first-rejection + residual distribution.

    Parameters
    ----------
    lp_p : torch.Tensor
        Target log-probs. Shape: ``(B, γ, V)``, float32.
    lp_q : torch.Tensor
        Draft log-probs.  Shape: ``(B, γ, V)``, float32.
    draft_token_ids : torch.Tensor
        Draft token indices. Shape: ``(B, γ)``, int64.
    u : torch.Tensor
        Uniform samples. Shape: ``(B, γ)``, float32 in [0, 1).

    Returns
    -------
    accepted_mask : torch.Tensor
        Boolean acceptance mask. Shape: ``(B, γ)``.
    first_rejection : torch.Tensor
        First rejection index per sequence. Shape: ``(B,)``, int32.
    residual : torch.Tensor
        Residual distributions. Shape: ``(B, V)``, float32.
    """
    batch_size, gamma, vocab_size = lp_p.shape

    # Gather log-probs at draft token positions
    ids = draft_token_ids.unsqueeze(-1)             # (B, γ, 1)
    lp_q_tok = lp_q.gather(-1, ids).squeeze(-1)    # (B, γ)
    lp_p_tok = lp_p.gather(-1, ids).squeeze(-1)    # (B, γ)

    # Acceptance criterion: u < min(1, exp(log_p - log_q))
    log_ratio = lp_p_tok - lp_q_tok
    accept_prob = log_ratio.exp().clamp(max=1.0)
    accepted_mask = u < accept_prob                 # (B, γ) bool

    # First rejection
    rejected = ~accepted_mask
    has_rejection = rejected.any(dim=-1)
    first_rej_raw = rejected.int().argmax(dim=-1)
    first_rejection = torch.where(
        has_rejection,
        first_rej_raw,
        torch.full_like(first_rej_raw, gamma),
    ).to(torch.int32)

    # Residual distribution
    residual = residual_dist_batched(lp_p, lp_q, first_rejection)

    return accepted_mask, first_rejection, residual
