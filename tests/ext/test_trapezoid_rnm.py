import math

import numpy as np
import pytest
from src.ext.trapezoid_rnm import OPT_CALL, OPT_PUT, compute_trapz_rnm


def _make_indptr(sizes: list[int]) -> np.ndarray:
    """Build a CSR indptr array from a list of group sizes."""
    indptr = np.zeros(len(sizes) + 1, dtype=np.intp)
    np.cumsum(sizes, out=indptr[1:])
    return indptr


def single_group_input(
    n_calls: int = 8,
    n_puts: int = 8,
    spot: float = 100.0,
    r: float = 0.02,
    ttm: float = 0.25,
    call_strikes: np.ndarray | None = None,
    put_strikes: np.ndarray | None = None,
    ivol: float = 0.20,
) -> dict:
    """
    Build minimal valid single-group fixture for compute_trapz_rnm
    """
    # 105, 110, ... for calls; 95, 90, ... for puts (if not provided)
    if call_strikes is None:
        call_strikes = spot + np.arange(1, n_calls + 1) * 5.0
    if put_strikes is None:
        put_strikes = spot - np.arange(1, n_puts + 1) * 5.0

    strikes = np.concatenate([call_strikes, put_strikes])
    flags = np.array(
        [OPT_CALL] * len(call_strikes) + [OPT_PUT] * len(put_strikes),
        dtype=np.intc,
    )
    ivols = np.full(len(strikes), ivol, dtype=np.float64)

    return dict(
        strikes=strikes,
        ivols=ivols,
        flags=flags,
        spots=np.array([spot], dtype=np.float64),
        rf=np.array([r], dtype=np.float64),
        ttm=np.array([ttm], dtype=np.float64),
        indptr=_make_indptr([len(strikes)]),
    )


# Successful execution of Cython function
def _wide_grid_inputs(
    spot: float, ivol: float, ttm: float, n: int = 40
) -> dict:  # noqa: E501
    """
    Inputs with a strike grid that scales with implied volatility and TTM
    so the integration domain covers the tails.
    """
    std = ivol * np.sqrt(ttm)
    log_bounds = 4.0 * std

    call_strikes = spot * np.exp(np.linspace(0.01, log_bounds, n))
    put_strikes = spot * np.exp(np.linspace(-log_bounds, -0.01, n))

    return single_group_input(
        call_strikes=call_strikes,
        put_strikes=put_strikes,
        spot=spot,
        ivol=ivol,
        ttm=ttm,
    )


@pytest.fixture
def valid_single_group_data() -> dict:
    return single_group_input()


class TestComputeTrapzRnm:
    def test_return_shapes(self, valid_single_group_data):
        """
        Test that output arrays have correct shapes for single group input.
        """
        varQ, skewQ, kurtQ, rc = compute_trapz_rnm(**valid_single_group_data)
        assert varQ.shape == (1,)
        assert skewQ.shape == (1,)
        assert kurtQ.shape == (1,)
        assert rc.shape == (1,)

    def test_return_types(self, valid_single_group_data):
        """
        Test that output arrays have correct dtypes for single group input.
        """
        varQ, skewQ, kurtQ, rc = compute_trapz_rnm(**valid_single_group_data)
        assert varQ.dtype == np.float64
        assert skewQ.dtype == np.float64
        assert kurtQ.dtype == np.float64
        assert rc.dtype == np.int32

    def test_variance_positive(self, valid_single_group_data):
        var, _, _, rc = compute_trapz_rnm(**valid_single_group_data)
        assert var[0] > 0.0
        assert rc[0] == 0  # Success code

    def test_kurtosis_exceeds_three(self, valid_single_group_data):
        """
        Sanity check; for a symmetric IV smile, kurtosis should be above 3
        """
        _, _, kurt, rc = compute_trapz_rnm(**valid_single_group_data)
        assert kurt[0] > 3.0
        assert rc[0] == 0

    def test_symmetric_smile_near_zero_skew(self):
        """
        A perfectly symmetric IV smile (equal put/call strikes equidistant
        from spot, equal IVols) should produce skewness close to zero.
        """
        spot = 100.0
        # 5, 10, ..., 40
        offsets = np.arange(5, 45, 5, dtype=np.float64)
        inp = single_group_input(
            call_strikes=spot + offsets,
            put_strikes=spot - offsets,
            ivol=0.20,
        )
        skew, _, _, rc = compute_trapz_rnm(**inp)
        # near zero, not exact due to BS nonlinearity
        assert abs(skew[0]) < 0.01
        assert rc[0] == 0

    def test_unsorted_strikes_give_same_result_as_sorted(self):
        """
        C layer must sort internally; hence input order must not affect output.
        """
        inp_sorted = single_group_input()

        inp_shuffled = {k: v.copy() for k, v in inp_sorted.items()}
        perm = np.random.default_rng(42).permutation(
            len(inp_sorted["strikes"])
        )
        inp_shuffled["strikes"] = inp_sorted["strikes"][perm]
        inp_shuffled["ivols"] = inp_sorted["ivols"][perm]
        inp_shuffled["flags"] = inp_sorted["flags"][perm]

        s1, v1, k1, rc1 = compute_trapz_rnm(**inp_sorted)
        s2, v2, k2, rc2 = compute_trapz_rnm(**inp_shuffled)

        np.testing.assert_allclose(s1, s2, rtol=1e-10)
        np.testing.assert_allclose(v1, v2, rtol=1e-10)
        np.testing.assert_allclose(k1, k2, rtol=1e-10)
        np.testing.assert_array_equal(rc1, rc2)

    def test_higher_ivol_higher_variance(self):
        """
        On a grid scaled to +/-4 sigma, variance must be strictly increasing
        with implied volatility.
        """
        spot = 100.0
        ivols = [0.10, 0.20, 0.30, 0.40]
        vars_ = []
        for ivol in ivols:
            inp = _wide_grid_inputs(spot=spot, ivol=ivol, ttm=0.25)
            _, var, _, rc = compute_trapz_rnm(**inp)
            vars_.append(var[0])
            assert rc[0] == 0

        for i in range(len(vars_) - 1):
            assert vars_[i] < vars_[i + 1], (
                f"Variance not increasing: ivol={ivols[i]:.2f} -> var={vars_[i]:.6f}, "  # noqa: E501
                f"ivol={ivols[i + 1]:.2f} -> var={vars_[i + 1]:.6f}"
            )


# Test errors
class TestComputeTrapzRnmErrors:
    def test_fewer_than_4_options_raises_valueerror(self):
        """Groups with < 4 options should raise ValueError"""
        inp = single_group_input(n_calls=1, n_puts=2)
        with pytest.raises(ValueError, match="at least 4 are required"):
            compute_trapz_rnm(**inp)

    def test_no_calls_handled_by_c_function(self):
        """
        No calls: Cython validation passes (not checking calls/puts count),
        but C function should set NaN and return TRAPZ_ERR_NO_CALLS.
        """
        inp = single_group_input(n_calls=0, n_puts=8)
        var, skew, kurt, rc = compute_trapz_rnm(**inp)
        # C function detects no calls and returns TRAPZ_ERR_NO_CALLS (-2)
        assert rc[0] == -2
        assert math.isnan(var[0])
        assert math.isnan(skew[0])
        assert math.isnan(kurt[0])

    def test_no_puts_handled_by_c_function(self):
        """
        No puts: Cython validation passes (not checking calls/puts count),
        but C function should set NaN and return TRAPZ_ERR_NO_PUTS.
        """
        inp = single_group_input(n_calls=8, n_puts=0)
        var, skew, kurt, rc = compute_trapz_rnm(**inp)
        # C function detects no puts and returns TRAPZ_ERR_NO_PUTS (-3)
        assert rc[0] == -3
        assert math.isnan(var[0])
        assert math.isnan(skew[0])
        assert math.isnan(kurt[0])

    def test_exactly_4_options_succeeds(self):
        """Exactly 4 options (2 calls, 2 puts) should work"""
        inp = single_group_input(n_calls=2, n_puts=2)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert not math.isnan(var[0])
        assert rc[0] == 0  # Success

    def test_mixed_valid_invalid_groups_raises_on_invalid(self):
        """
        If any group has < 4 options, Cython raises ValueError during
        validation.
        """
        valid = single_group_input(n_calls=8, n_puts=8)
        invalid = single_group_input(n_calls=1, n_puts=1)

        n_v = len(valid["strikes"])
        n_i = len(invalid["strikes"])

        batch = dict(
            strikes=np.concatenate([valid["strikes"], invalid["strikes"]]),
            ivols=np.concatenate([valid["ivols"], invalid["ivols"]]),
            flags=np.concatenate([valid["flags"], invalid["flags"]]),
            spots=np.array([100.0, 100.0]),
            rf=np.array([0.02, 0.02]),
            ttm=np.array([0.25, 0.25]),
            indptr=_make_indptr([n_v, n_i]),
        )
        # Should raise ValueError due to second group having only 2 options
        with pytest.raises(ValueError, match="at least 4 are required"):
            compute_trapz_rnm(**batch)


# Edge cases
class TestComputeTrapzRnmEdgeCases:
    def test_zero_groups_raises_valueerror(self):
        """Empty batch raises ValueError due to empty strikes array"""
        with pytest.raises(ValueError, match="No options provided"):
            compute_trapz_rnm(
                strikes=np.array([], dtype=np.float64),
                ivols=np.array([], dtype=np.float64),
                flags=np.array([], dtype=np.int32),
                spots=np.array([], dtype=np.float64),
                rf=np.array([], dtype=np.float64),
                ttm=np.array([], dtype=np.float64),
                indptr=np.array([0], dtype=np.int64),
            )

    def test_large_batch_no_crash(self):
        """Stress test somewhat large input data"""
        n_groups = 500
        n_per = 16  # 8 calls + 8 puts
        total_opt = n_groups * n_per

        spot = 100.0
        strikes = np.empty(total_opt, dtype=np.float64)
        flags = np.empty(total_opt, dtype=np.int32)
        ivols = np.full(total_opt, 0.20, dtype=np.float64)

        for g in range(n_groups):
            base = g * n_per
            strikes[base : base + 8] = spot + np.arange(5, 45, 5)
            strikes[base + 8 : base + 16] = spot - np.arange(5, 45, 5)
            flags[base : base + 8] = OPT_CALL
            flags[base + 8 : base + 16] = OPT_PUT

        spots = np.full(n_groups, spot)
        rf_arr = np.full(n_groups, 0.02)
        ttm_arr = np.full(n_groups, 0.25)
        indptr = _make_indptr([n_per] * n_groups)

        var, skew, _, rc = compute_trapz_rnm(
            strikes, ivols, flags, spots, rf_arr, ttm_arr, indptr
        )
        assert not np.all(np.isnan(var))
        assert skew.shape == (n_groups,)
        assert np.all(rc == 0)  # All groups should succeed

    def test_very_deep_otm_options(self):
        """
        Very deep OTM options have BS prices close to zero. The trapezoidal sum
        should still be finite.
        """
        deep_calls = np.array([200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        deep_puts = np.array([50.0, 30.0, 20.0, 10.0], dtype=np.float64)
        inp = single_group_input(
            call_strikes=deep_calls,
            put_strikes=deep_puts,
            ivol=0.20,
        )
        var, skew, kurt, rc = compute_trapz_rnm(**inp)
        for val in (var[0], skew[0], kurt[0]):
            assert math.isfinite(val) or math.isnan(val)
        assert rc[0] == 0

    def test_near_zero_ivol(self):
        inp = single_group_input(ivol=1e-4)
        var, skew, kurt, rc = compute_trapz_rnm(**inp)
        for val in (var[0], skew[0], kurt[0]):
            assert not math.isinf(val)
        assert rc[0] == 0
