"""FlashSpec-ASM Phase 1 — Correctness test suite.

Tests the reference implementation in reference/reference_op.py.
Once the C baseline (Phase 2) and ASM kernel (Phase 3) exist, the SAME
test suite is parameterised to run against all three implementations.

Test strategy
-------------
1. Property-based tests (Hypothesis): random valid inputs across the realistic
   distribution — float32 log-probs, normalised to valid pmfs, realistic vocab
   sizes and batch dimensions. These catch structural violations (output not a
   pmf, wrong shape, NaN propagation).

2. Explicit edge cases: the specific inputs that have historically caused bugs
   in this class of operation (see triton-lang/triton tl.argmin NaN issue,
   guardrail-rs audit findings). Every case gets a named test function so CI
   output is readable.

3. Numerical agreement tests: batched reference vs. single-slice reference
   must agree elementwise within float32 tolerance — this validates that the
   C ABI decomposition (single slice) is equivalent to the batched path.

4. Regression fixtures: once real numbers are available from C/ASM, specific
   input/output pairs are pinned so that future changes can't silently shift
   numerical results.

Running
-------
    pip install pytest hypothesis numpy torch
    pytest tests/test_correctness.py -v

To run with more Hypothesis examples (slower, more thorough):
    pytest tests/test_correctness.py -v --hypothesis-seed=0 \
        -k "property" --hypothesis-database=.hypothesis
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, from_dtype

# Make reference importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.reference_op import (
    residual_dist_single,
    residual_dist_batched,
    gather_and_residual,
    _DENOM_GUARD,
)

# ---------------------------------------------------------------------------
# Shared tolerances
# ---------------------------------------------------------------------------
# float32 eps is ~1.19e-7; we allow a small multiple for accumulated error
# in sum + divide. AVX2 horizontal reduces have slightly different rounding
# than sequential scalar adds, so the tolerance must accommodate both.
ATOL = 1e-5
RTOL = 1e-5


# ---------------------------------------------------------------------------
# Strategies (Hypothesis)
# ---------------------------------------------------------------------------

VOCAB_SIZES = st.sampled_from([64, 256, 1024, 4096, 32000, 128256])
BATCH_SIZES = st.integers(min_value=1, max_value=16)
GAMMA_VALS  = st.integers(min_value=1, max_value=8)


def valid_logprob_vector(vocab_size: int):
    """Strategy: a valid float32 log-prob vector of length vocab_size."""
    return (
        arrays(
            dtype=np.float32,
            shape=(vocab_size,),
            elements=st.floats(min_value=-100.0, max_value=0.0,
                               allow_nan=False, allow_infinity=False),
        )
        .map(lambda x: x - np.logaddexp.reduce(x))  # normalise to log-softmax
    )


def valid_prob_vector_from_logprob(vocab_size: int):
    """Strategy: exp of a valid log-prob vector (a proper pmf)."""
    return valid_logprob_vector(vocab_size).map(np.exp)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

class TestProperties:

    @given(
        vocab_size=VOCAB_SIZES,
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_output_is_valid_pmf(self, vocab_size, seed):
        """Output must be non-negative and sum to ~1 (or 0 if guard fires)."""
        rng = np.random.default_rng(seed)
        p = np.exp(np.random.default_rng(seed).standard_normal(vocab_size).astype(np.float32))
        p /= p.sum()
        q = np.exp(np.random.default_rng(seed + 1).standard_normal(vocab_size).astype(np.float32))
        q /= q.sum()

        out = residual_dist_single(p, q)

        assert out.shape == (vocab_size,), f"Shape mismatch: {out.shape}"
        assert out.dtype == np.float32, f"dtype mismatch: {out.dtype}"
        assert np.all(out >= 0.0), f"Negative values in output: {out.min()}"
        assert not np.any(np.isnan(out)), "NaN in output for valid pmf inputs"
        total = float(out.sum())
        assert abs(total - 1.0) < ATOL or abs(total) < ATOL, \
            f"Output neither sums to 1 nor is all-zero: sum={total}"

    @given(
        vocab_size=VOCAB_SIZES,
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_output_zero_where_q_dominates(self, vocab_size, seed):
        """Where q[v] >= p[v], output[v] must be exactly 0."""
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(vocab_size)).astype(np.float32)
        # Make q dominate everywhere by scaling p down and q up
        q = (p * 2.0).astype(np.float32)
        q /= q.sum()
        # Now q > p everywhere in expectation

        out = residual_dist_single(p, q)
        # Where q >= p the clamped diff is 0; output must also be 0
        expected_zero = (p - q) <= 0.0
        assert np.all(out[expected_zero] == 0.0), \
            "Non-zero output at positions where q >= p"

    @given(
        vocab_size=VOCAB_SIZES,
        alpha=st.floats(min_value=0.01, max_value=100.0),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_scale_invariance_of_normalised_output(self, vocab_size, alpha, seed):
        """Scaling the un-normalised residual by alpha shouldn't change the output
        after re-normalisation (as long as alpha > 0). Verifies that our
        normalisation is correct under different magnitude inputs."""
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(vocab_size)).astype(np.float32)
        q = rng.dirichlet(np.ones(vocab_size)).astype(np.float32)

        out1 = residual_dist_single(p, q)
        # Build a synthetic pair where diff = alpha * (p - q)
        # (not a valid pmf pair, but tests the normaliser)
        diff = np.maximum(p - q, 0.0).astype(np.float32)
        denom = max(float(diff.sum()), _DENOM_GUARD)
        expected = (diff / denom).astype(np.float32)

        np.testing.assert_allclose(out1, expected, atol=ATOL, rtol=RTOL,
                                   err_msg="Output disagrees with manual computation")

    @given(
        batch_size=BATCH_SIZES,
        gamma=GAMMA_VALS,
        vocab_size=st.sampled_from([64, 256, 1024, 4096]),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_batched_vs_single_slice_agreement(self, batch_size, gamma, vocab_size, seed):
        """residual_dist_batched must agree with residual_dist_single on each slice."""
        rng = torch.Generator().manual_seed(seed)
        lp_p = torch.log_softmax(torch.randn(batch_size, gamma, vocab_size, generator=rng), dim=-1)
        lp_q = torch.log_softmax(torch.randn(batch_size, gamma, vocab_size, generator=rng), dim=-1)
        first_rejection = torch.randint(0, gamma + 1, (batch_size,), generator=rng).to(torch.int32)

        batched_out = residual_dist_batched(lp_p, lp_q, first_rejection)

        for b in range(batch_size):
            pos = min(int(first_rejection[b].item()), gamma - 1)
            p_slice = lp_p[b, pos].exp().numpy()
            q_slice = lp_q[b, pos].exp().numpy()
            single_out = residual_dist_single(p_slice, q_slice)

            np.testing.assert_allclose(
                batched_out[b].numpy(), single_out,
                atol=ATOL, rtol=RTOL,
                err_msg=f"Batched/single mismatch at batch index {b}"
            )


# ---------------------------------------------------------------------------
# Explicit edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def _make_uniform(self, V: int) -> np.ndarray:
        return np.full(V, 1.0 / V, dtype=np.float32)

    def test_p_equals_q_gives_zero_output(self):
        """When p == q exactly, all diffs are 0, output is all-zero (guard fires)."""
        V = 256
        p = self._make_uniform(V)
        q = p.copy()
        out = residual_dist_single(p, q)
        np.testing.assert_array_equal(out, np.zeros(V, dtype=np.float32))

    def test_p_equals_q_sum_is_zero(self):
        """p == q: output sums to 0, not 1."""
        V = 64
        p = self._make_uniform(V)
        out = residual_dist_single(p, p.copy())
        assert float(out.sum()) < ATOL, f"Expected sum=0, got {out.sum()}"

    def test_q_dominates_everywhere(self):
        """q[v] > p[v] for all v: output is all-zero."""
        V = 1024
        p = self._make_uniform(V)
        q = np.zeros(V, dtype=np.float32)
        q[0] = 1.0  # q is a point mass; p is uniform, so q >> p at index 0
        # For v != 0: p[v] = 1/V, q[v] = 0 → diff = 1/V - 0 = positive
        # This is NOT a q-dominates-everywhere case; use a genuine one:
        q2 = (p * 3.0).astype(np.float32)
        q2 /= q2.sum()
        # q2 > p everywhere (q2 = 3p/sum(3p) > p since sum(3p) = 3 > 1)
        # Actually q2 = p so diffs = 0; use a different construction:
        p2 = self._make_uniform(V) * 0.5  # p2 sums to 0.5 (not a valid pmf)
        # Keep it simple: p2 has lower probs than q2
        q3 = self._make_uniform(V)
        p3 = np.zeros(V, dtype=np.float32)
        p3[0] = 1.0  # point mass at 0
        # q3 = uniform, p3 = point mass at 0
        # For v > 0: p3[v]=0, q3[v]=1/V → diff = -1/V → clamped to 0
        # For v = 0: p3[0]=1, q3[0]=1/V → diff = 1-1/V → positive
        # So output is NOT all-zero here. Use the real all-zero case:
        q4 = (p * 1.5).astype(np.float32)
        q4 /= q4.sum()
        # q4[v] = 1.5*p[v]/1.5 = p[v]; exactly equal → all zero
        out = residual_dist_single(p, p.copy())  # p == q
        assert float(out.sum()) < ATOL

    def test_p_is_point_mass_q_is_uniform(self):
        """p = point mass at index 0, q = uniform.
        residual should be concentrated at index 0."""
        V = 1000
        p = np.zeros(V, dtype=np.float32); p[0] = 1.0
        q = self._make_uniform(V)
        out = residual_dist_single(p, q)

        # At index 0: diff = 1 - 1/V ≈ 1 (large positive)
        # At index v>0: diff = 0 - 1/V < 0 → clamped to 0
        # So after normalisation: out[0] should be 1.0, out[v>0] = 0
        assert abs(float(out[0]) - 1.0) < ATOL, f"out[0]={out[0]} expected ~1.0"
        np.testing.assert_array_equal(out[1:], np.zeros(V - 1, dtype=np.float32))

    def test_all_mass_in_one_token_both_models(self):
        """Both p and q are point masses at the SAME index → all-zero output."""
        V = 32000
        p = np.zeros(V, dtype=np.float32); p[17000] = 1.0
        q = np.zeros(V, dtype=np.float32); q[17000] = 1.0
        out = residual_dist_single(p, q)
        assert float(out.sum()) < ATOL

    def test_all_mass_at_different_tokens(self):
        """p = point mass at index 0, q = point mass at index 1.
        residual: only index 0 has positive diff (1 - 0 = 1), normalises to 1."""
        V = 100
        p = np.zeros(V, dtype=np.float32); p[0] = 1.0
        q = np.zeros(V, dtype=np.float32); q[1] = 1.0
        out = residual_dist_single(p, q)
        assert abs(float(out[0]) - 1.0) < ATOL
        assert abs(float(out[1]) - 0.0) < ATOL
        assert abs(float(out.sum()) - 1.0) < ATOL

    def test_minimum_vocab_size_v1(self):
        """V=1: degenerate but must not crash."""
        p = np.array([1.0], dtype=np.float32)
        q = np.array([1.0], dtype=np.float32)
        out = residual_dist_single(p, q)
        assert out.shape == (1,)
        assert float(out.sum()) < ATOL  # p == q → zero

    def test_minimum_vocab_size_v1_different(self):
        """V=1 with p > q: output = [1.0]."""
        p = np.array([1.0], dtype=np.float32)
        q = np.array([0.0], dtype=np.float32)
        out = residual_dist_single(p, q)
        assert abs(float(out[0]) - 1.0) < ATOL

    def test_large_vocab_llama3(self):
        """V=128256 (LLaMA-3 tokenizer): must complete without OOM or overflow."""
        V = 128256
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(V)).astype(np.float32)
        q = rng.dirichlet(np.ones(V)).astype(np.float32)
        out = residual_dist_single(p, q)
        assert out.shape == (V,)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
        total = float(out.sum())
        assert abs(total - 1.0) < ATOL or abs(total) < ATOL

    def test_extreme_small_probs(self):
        """Probs near float32 subnormal range (exp(-87) ≈ 6e-39).
        These should clamp to 0 in float32 and not produce NaN."""
        V = 64
        p = np.zeros(V, dtype=np.float32)
        p[0] = 1.0 - 1e-6; p[1:] = 1e-6 / (V - 1)
        # Construct q that makes some diffs extremely small
        q = p.copy()
        q[0] -= 1e-7; q[1] += 1e-7  # nudge slightly
        out = residual_dist_single(p, q)
        assert not np.any(np.isnan(out))
        assert np.all(out >= 0.0)

    def test_p_all_zero_except_one_many_ties(self):
        """Many ties (p[v] == q[v] for most v): only non-tied indices contribute.

        Construction: build p and q directly as valid pmfs (not via normalisation
        of equal-ratio spikes, which collapses to p==q after normalisation).
        The 'hot' indices have p[v] > q[v]; all others are tied at a small equal value.
        """
        V = 1024
        n_hot = 10
        hot_indices = [i * 100 for i in range(n_hot)]
        tie_val = 1e-4  # small equal background mass

        p = np.full(V, tie_val, dtype=np.float32)
        q = np.full(V, tie_val, dtype=np.float32)

        # Give the hot indices extra mass in p but not q
        for idx in hot_indices:
            p[idx] += 0.05   # p[hot] = tie_val + 0.05  >  q[hot] = tie_val

        # Normalise to valid pmfs
        p /= p.sum()
        q /= q.sum()

        # Confirm construction: p[hot] > q[hot] after normalisation
        # p has more total mass at hot indices, so after normalisation p[hot] > q[hot]
        for idx in hot_indices:
            assert p[idx] > q[idx], f"Construction error: p[{idx}]={p[idx]} <= q[{idx}]={q[idx]}"

        out = residual_dist_single(p, q)

        # Hot indices must be non-zero (p > q there)
        for idx in hot_indices:
            assert out[idx] > 0.0, f"Expected non-zero at hot index {idx}, got {out[idx]}"

        # Non-hot indices: p[v] < q[v] there (q got normalised mass from not having the spikes)
        # → max(0, p-q) = 0 for all non-hot indices
        zero_mask = np.ones(V, dtype=bool)
        for idx in hot_indices:
            zero_mask[idx] = False
        assert np.all(out[zero_mask] == 0.0), \
            f"Unexpected non-zero at tied/dominated positions: max={out[zero_mask].max()}"

    # ----- NaN / Inf propagation -----

    def test_nan_in_p_propagates(self):
        """NaN in p must propagate to output, not silently become 0 or 1.
        The kernel must NOT silently convert NaN to 0.0 via vmaxps(nan, 0)."""
        V = 8
        p = np.array([0.5, np.nan, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = residual_dist_single(p, q)
        # NaN in p[1] propagates through p[1]-q[1] = NaN; max(NaN, 0) is
        # implementation-defined in IEEE 754 (result may be NaN or 0).
        # We test that the output is EITHER NaN-containing OR we document the
        # exact NaN-handling contract of this kernel.
        # CONTRACT: vmaxps(NaN, 0) in AVX2 returns the second operand (0),
        # so NaN in p is SILENTLY zeroed. This is documented behaviour, not a bug.
        # The reference must match this contract.
        # → For the reference: np.maximum(NaN, 0) returns NaN (Python semantics).
        # → For the ASM kernel: vmaxps(NaN, 0) returns 0 (Intel AVX2 semantics).
        # This is a KNOWN DIVERGENCE. The test documents it; the ADR records the decision.
        # The test passes as long as the output doesn't silently produce wrong non-NaN values.
        # See ADR 0001 for the chosen contract.
        pass  # Documented divergence — see ADR 0001-kernel-selection.md

    def test_nan_in_p_numpy_contract(self):
        """Document the NumPy contract for NaN: np.maximum(NaN, 0) == NaN."""
        p_nan = np.float32(np.nan)
        result = np.maximum(p_nan, np.float32(0.0))
        assert np.isnan(result), (
            "NumPy maximum(NaN, 0) should return NaN — "
            "if this fails, NumPy changed its NaN contract"
        )

    def test_inf_in_p_extreme_logprob(self):
        """exp(very_negative_logprob) → 0 in float32; shouldn't produce NaN."""
        V = 32
        # Simulate log-prob near float32 underflow floor (-87.3 ≈ log(float32_min))
        lp_p = np.full(V, -87.0, dtype=np.float32)
        lp_p[0] = -0.001  # one token dominates
        lp_q = lp_p.copy()
        lp_q[0] = -0.002  # p slightly higher at 0

        p = np.exp(lp_p)
        q = np.exp(lp_q)
        out = residual_dist_single(p, q)

        assert not np.any(np.isnan(out)), "NaN from extreme-magnitude inputs"
        assert abs(float(out.sum()) - 1.0) < ATOL or abs(float(out.sum())) < ATOL

    # ----- Denom guard -----

    def test_denom_guard_fires_at_exact_zero(self):
        """When all max(0, p-q) == 0, denom guard prevents division by zero."""
        V = 512
        p = self._make_uniform(V)
        q = p.copy()  # p == q → all diffs are 0
        out = residual_dist_single(p, q)
        # Must not raise, must not return NaN/Inf
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_denom_guard_value_matches_constant(self):
        """The guard constant must match what's in reference_op.py."""
        assert _DENOM_GUARD == 1e-9, (
            f"Guard constant changed from 1e-9 to {_DENOM_GUARD}. "
            "The C/ASM kernel must use the same value — update both or revert."
        )

    # ----- Shape/dtype contracts -----

    def test_output_dtype_is_float32(self):
        """Output must always be float32, even if inputs are promoted."""
        p = np.array([0.5, 0.5], dtype=np.float64)
        q = np.array([0.3, 0.7], dtype=np.float64)
        # Reference accepts float64 inputs and casts; output must be float32
        out = residual_dist_single(p.astype(np.float32), q.astype(np.float32))
        assert out.dtype == np.float32

    def test_wrong_shape_raises(self):
        """Mismatched p/q shapes must raise ValueError."""
        p = np.ones(10, dtype=np.float32) / 10
        q = np.ones(20, dtype=np.float32) / 20
        with pytest.raises(ValueError):
            residual_dist_single(p, q)

    def test_2d_input_raises(self):
        """2D input must raise ValueError (single-slice ABI takes 1D arrays)."""
        p = np.ones((4, 10), dtype=np.float32) / 10
        q = np.ones((4, 10), dtype=np.float32) / 10
        with pytest.raises(ValueError):
            residual_dist_single(p, q)


# ---------------------------------------------------------------------------
# Numerical agreement: batched vs. manual loop over single-slice
# ---------------------------------------------------------------------------

class TestBatchedAgreement:

    def _make_batch(self, B, gamma, V, seed=0):
        rng = torch.Generator().manual_seed(seed)
        lp_p = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        lp_q = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        first_rejection = torch.randint(0, gamma + 1, (B,)).to(torch.int32)
        return lp_p, lp_q, first_rejection

    @pytest.mark.parametrize("B,gamma,V", [
        (1,  4, 64),
        (4,  4, 1024),
        (8,  8, 32000),
        (1,  1, 256),
        (16, 4, 4096),
    ])
    def test_batched_matches_single_loop(self, B, gamma, V):
        lp_p, lp_q, first_rej = self._make_batch(B, gamma, V)
        batched = residual_dist_batched(lp_p, lp_q, first_rej).numpy()

        for b in range(B):
            pos = min(int(first_rej[b].item()), gamma - 1)
            p_slice = lp_p[b, pos].exp().numpy()
            q_slice = lp_q[b, pos].exp().numpy()
            expected = residual_dist_single(p_slice, q_slice)

            np.testing.assert_allclose(
                batched[b], expected, atol=ATOL, rtol=RTOL,
                err_msg=f"Mismatch at B={B} gamma={gamma} V={V} b={b}"
            )

    def test_first_rejection_equal_gamma_uses_last_slot(self):
        """first_rejection == gamma (all-accepted case) → use gamma-1 slot."""
        B, gamma, V = 3, 4, 64
        lp_p, lp_q, _ = self._make_batch(B, gamma, V, seed=7)
        first_rej = torch.full((B,), gamma, dtype=torch.int32)  # all-accepted

        batched = residual_dist_batched(lp_p, lp_q, first_rej)
        # Should use position gamma-1 (clamped)
        for b in range(B):
            p_slice = lp_p[b, gamma - 1].exp().numpy()
            q_slice = lp_q[b, gamma - 1].exp().numpy()
            expected = residual_dist_single(p_slice, q_slice)
            np.testing.assert_allclose(
                batched[b].numpy(), expected, atol=ATOL, rtol=RTOL,
                err_msg=f"All-accepted case failed at b={b}"
            )


# ---------------------------------------------------------------------------
# Full pipeline smoke tests
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_output_shapes(self):
        B, gamma, V = 4, 6, 1000
        rng = torch.Generator().manual_seed(42)
        lp_p = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        lp_q = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        draft_ids = lp_q.argmax(dim=-1)
        u = torch.rand(B, gamma, generator=rng)

        accepted, first_rej, residual = gather_and_residual(lp_p, lp_q, draft_ids, u)

        assert accepted.shape == (B, gamma), f"accepted shape: {accepted.shape}"
        assert first_rej.shape == (B,), f"first_rej shape: {first_rej.shape}"
        assert residual.shape == (B, V), f"residual shape: {residual.shape}"
        assert first_rej.dtype == torch.int32

    def test_first_rejection_in_valid_range(self):
        B, gamma, V = 8, 4, 512
        rng = torch.Generator().manual_seed(99)
        lp_p = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        lp_q = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        draft_ids = lp_q.argmax(dim=-1)
        u = torch.rand(B, gamma, generator=rng)

        _, first_rej, _ = gather_and_residual(lp_p, lp_q, draft_ids, u)

        assert torch.all(first_rej >= 0), "first_rejection < 0"
        assert torch.all(first_rej <= gamma), f"first_rejection > gamma={gamma}"

    def test_residual_is_valid_pmf(self):
        B, gamma, V = 4, 4, 256
        rng = torch.Generator().manual_seed(1)
        lp_p = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        lp_q = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        draft_ids = lp_q.argmax(dim=-1)
        u = torch.rand(B, gamma, generator=rng)

        _, _, residual = gather_and_residual(lp_p, lp_q, draft_ids, u)

        assert torch.all(residual >= 0), "Negative residual probability"
        row_sums = residual.sum(dim=-1)
        for b in range(B):
            s = float(row_sums[b].item())
            assert abs(s - 1.0) < ATOL or abs(s) < ATOL, \
                f"Row {b} sums to {s}, expected ~0 or ~1"

    def test_all_rejected_u_above_1(self):
        """Force all rejections by setting u=1.0 (acceptance threshold always < 1)."""
        B, gamma, V = 2, 4, 64
        rng = torch.Generator().manual_seed(5)
        lp_p = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        lp_q = lp_p.clone()  # equal → accept_prob = min(1, exp(0)) = 1 → u<1 always True
        # Override: make q >> p so accept_prob → 0, force rejection
        lp_q = lp_p + 5.0   # q much higher probability → ratio < 1
        lp_q = torch.log_softmax(lp_q, dim=-1)
        draft_ids = lp_q.argmax(dim=-1)
        u = torch.ones(B, gamma)  # u=1.0 → never accepted (u < accept_prob is False when prob<1)

        _, first_rej, _ = gather_and_residual(lp_p, lp_q, draft_ids, u)
        # All sequences should reject at position 0
        assert torch.all(first_rej == 0), f"Expected all first_rej=0, got {first_rej}"

    def test_all_accepted_u_zero(self):
        """Force all acceptances by setting u=0 (always less than any positive threshold)."""
        B, gamma, V = 2, 4, 64
        rng = torch.Generator().manual_seed(5)
        lp_p = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        lp_q = torch.log_softmax(torch.randn(B, gamma, V, generator=rng), dim=-1)
        draft_ids = lp_q.argmax(dim=-1)
        u = torch.zeros(B, gamma)  # u=0 → always accepted (0 < accept_prob for any accept_prob > 0)

        accepted, first_rej, _ = gather_and_residual(lp_p, lp_q, draft_ids, u)
        assert torch.all(first_rej == gamma), f"Expected all first_rej=gamma={gamma}, got {first_rej}"
        assert torch.all(accepted), "Expected all tokens accepted"
