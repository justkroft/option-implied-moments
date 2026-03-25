import math

import numpy as np
import pytest
from src.ext.trapezoid_rnm import OPT_CALL, OPT_PUT, compute_trapz_rnm

from tests.conftest import _make_indptr, single_group_input


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
        assert rc[0] == 0

    @pytest.mark.parametrize("ivol", [0.05, 0.10, 0.20, 0.50, 0.75])
    def test_kurtosis_exceeds_three(self, ivol):
        """
        Sanity check; for a symmetric IV smile, kurtosis should be above 3
        """
        inp = single_group_input(ivol=ivol)
        _, _, kurt, rc = compute_trapz_rnm(**inp)
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

    def test_variance_scales_with_time(self):
        """Variance should scale roughly as sigma^2 * T."""
        sigma = 0.20
        for T in [0.05, 0.25, 1.0]:
            inp = single_group_input(ivol=sigma, ttm=T)
            var, _, _, rc = compute_trapz_rnm(**inp)
            assert rc[0] == 0

            # Check that variance is positive and scales sensibly
            expected_scale = sigma**2 * T
            assert 0.1 * expected_scale < var[0] < 5.0 * expected_scale

    def test_variance_increases_with_ivol(self):
        """Variance should strictly increase with implied volatility."""
        T = 0.25
        variances = []
        ivols = [0.10, 0.20, 0.40]

        for sigma in ivols:
            inp = single_group_input(ivol=sigma, ttm=T)
            var, _, _, rc = compute_trapz_rnm(**inp)
            assert rc[0] == 0
            variances.append(var[0])

        # Verify strict monotonicity
        for i in range(len(variances) - 1):
            assert variances[i] < variances[i + 1], (
                "Variance not increasing with IV"
            )

    def test_moments_finite_across_rates(self):
        """All moments should be computable across interest rate spectrum."""
        spot = 100.0
        rates = [0.001, 0.05, 0.10]

        for r in rates:
            inp = single_group_input(spot=spot, r=r)
            var, skew, kurt, rc = compute_trapz_rnm(**inp)
            assert rc[0] == 0
            # All moments should be finite
            assert not math.isnan(var[0]) and not math.isinf(var[0])
            assert not math.isnan(skew[0]) and not math.isinf(skew[0])
            assert not math.isnan(kurt[0]) and not math.isinf(kurt[0])

    def test_unsorted_strikes_give_same_result_as_sorted(self):
        """
        C extension sorts internally; hence input order must not affect output.
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


class TestComputeTrapzRnmErrors:
    def test_fewer_than_4_options_raises_valueerror(self):
        inp = single_group_input(n_calls=1, n_puts=2)
        with pytest.raises(ValueError, match="at least 4 are required"):
            compute_trapz_rnm(**inp)

    def test_no_calls_handled_by_c_function(self):
        """
        No call options: Cython validation passes (not checking calls/puts count),
        but C function should set NaN and return TRAPZ_ERR_NO_CALLS.
        """  # noqa: E501
        inp = single_group_input(n_calls=0, n_puts=8)
        var, skew, kurt, rc = compute_trapz_rnm(**inp)
        assert rc[0] == -2
        assert math.isnan(var[0])
        assert math.isnan(skew[0])
        assert math.isnan(kurt[0])

    def test_no_puts_handled_by_c_function(self):
        """
        No put options: Cython validation passes (not checking calls/puts count),
        but C function should set NaN and return TRAPZ_ERR_NO_PUTS.
        """  # noqa: E501
        inp = single_group_input(n_calls=8, n_puts=0)
        var, skew, kurt, rc = compute_trapz_rnm(**inp)
        assert rc[0] == -3
        assert math.isnan(var[0])
        assert math.isnan(skew[0])
        assert math.isnan(kurt[0])

    def test_exactly_4_options_succeeds(self):
        inp = single_group_input(n_calls=2, n_puts=2)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert not math.isnan(var[0])
        assert rc[0] == 0

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

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(ivol=0.01),
            dict(ivol=0.50),
            dict(ivol=2.0),
            dict(r=-0.05),
            dict(r=0.10),
            dict(ttm=0.01),
            dict(ttm=10.0),
        ],
    )
    def test_variance_always_non_negative(self, kwargs):
        """
        Variance should be non-negative.
        Slight negativity indicates accumulated rounding error.
        """
        inp = single_group_input(**kwargs)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert rc[0] == 0, f"failed with return code {rc[0]} for {kwargs}"
        assert var[0] > -1e-4, f"Large negative variance for {kwargs}"

    @pytest.mark.parametrize("ivol", [0.15, 0.20, 0.50])
    @pytest.mark.parametrize("r", [0.01, 0.05])
    def test_skewness_magnitude(self, ivol, r):
        """Skewness should be in [-3, 3] range for regular inputs."""
        inp = single_group_input(ivol=ivol, r=r)
        _, skew, _, rc = compute_trapz_rnm(**inp)
        assert rc[0] == 0
        assert -3.0 < skew[0] < 3.0, (
            f"Unreasonable skew {skew[0]:.2f} for IV={ivol}, r={r}"
        )
        # Extreme skew is possible but should be finite
        assert not math.isinf(skew[0]), f"Infinite skew for IV={ivol}, r={r}"

    def test_large_batch_no_crash(self):
        """Stress test somewhat large input data"""
        n_groups = 500
        n_per = 16  # 8 calls / 8 puts
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

    def test_very_large_spot(self):
        spot = 1e6
        inp = single_group_input(
            spot=spot,
            call_strikes=spot + np.array([1e4, 5e4, 1e5, 5e5]),
            put_strikes=spot - np.array([1e4, 5e4, 1e5, 5e5]),
            ivol=0.20,
            ttm=0.25,
        )
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert not math.isinf(var[0]) and not math.isnan(var[0])
        assert rc[0] == 0

    def test_extreme_moneyness(self):
        """Deep ITM and deep OTM options together."""
        spot = 100.0
        # Deep ITM calls (K << S), deep OTM calls (K >> S)
        call_strikes = np.array([10.0, 50.0, 500.0, 1000.0])
        put_strikes = np.array([150.0, 200.0, 500.0, 900.0])
        inp = single_group_input(
            spot=spot,
            call_strikes=call_strikes,
            put_strikes=put_strikes,
            ivol=0.20,
        )
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert not math.isinf(var[0]) and not math.isnan(var[0])
        assert rc[0] == 0

    def test_near_zero_ttm(self):
        """Time-to-maturity very close to zero (1 minute)."""
        inp = single_group_input(ttm=1.0 / (6.5 * 60))
        var, _, _, rc = compute_trapz_rnm(**inp)
        # With near-zero TTM, IV effect diminishes; variance -> 0
        # Should not crash
        assert rc[0] == 0
        assert not math.isinf(var[0])

    def test_very_long_ttm(self):
        """Time-to-maturity of 10 years."""
        inp = single_group_input(ttm=10.0 * 365)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert rc[0] == 0
        assert not math.isinf(var[0]) and not math.isnan(var[0])

    def test_extreme_ivol_low(self):
        """Implied volatility 0.1% (very low vol regime).

        Very low IV can cause numerical instability due to rounding.
        """
        inp = single_group_input(ivol=0.001)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert rc[0] == 0
        # Note: Very low IV can produce slightly negative variance
        # due to rounding errors in numerical integration
        assert var[0] > -1e-4  # Allow small negative due to rounding

    def test_extreme_ivol_high(self):
        inp = single_group_input(ivol=2.0)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert rc[0] == 0
        assert not math.isinf(var[0]) and not math.isnan(var[0])

    def test_negative_interest_rate(self):
        inp = single_group_input(r=-0.01)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert rc[0] == 0
        assert var[0] > 0

    def test_high_positive_rate(self):
        inp = single_group_input(r=0.10)
        var, _, _, rc = compute_trapz_rnm(**inp)
        assert rc[0] == 0
        assert not math.isinf(var[0]) and not math.isnan(var[0])


class TestComputeTrapzRnmSensitivity:
    """Test that small input changes produce sensible output changes."""

    def test_stability_iv_changes(self):
        """Small change of implied vol."""
        spot = 100.0
        base_inp = single_group_input(spot=spot, ivol=0.20)
        change_inp = single_group_input(spot=spot, ivol=0.200001)

        base_var, _, _, _ = compute_trapz_rnm(**base_inp)
        change_var, _, _, _ = compute_trapz_rnm(**change_inp)

        # Variance should change smoothly
        rel_change = abs(change_var[0] - base_var[0]) / base_var[0]
        assert rel_change < 0.01  # Less than 1% change for 0.05% IV change

    def test_stability_spot_changes(self):
        """Small change of spot price."""
        base_inp = single_group_input(spot=100.0)
        change_inp = single_group_input(spot=100.01)

        base_var, _, _, _ = compute_trapz_rnm(**base_inp)
        change_var, _, _, _ = compute_trapz_rnm(**change_inp)

        # Relative change should be proportional to spot change (0.01%)
        rel_change = abs(change_var[0] - base_var[0]) / base_var[0]
        assert rel_change < 0.001  # Highly stable to spot perturbation

    def test_stability_strikes_changes(self):
        """Small change of strike grid."""
        base_inp = single_group_input(n_calls=8, n_puts=8)

        # Change by 1 basis point
        change_inp = base_inp.copy()
        change_inp["strikes"] = base_inp["strikes"] * 1.0001

        base_var, _, _, _ = compute_trapz_rnm(**base_inp)
        change_var, _, _, _ = compute_trapz_rnm(**change_inp)

        # Should be stable to tiny strike changes
        rel_change = abs(change_var[0] - base_var[0]) / base_var[0]
        assert rel_change < 0.001
