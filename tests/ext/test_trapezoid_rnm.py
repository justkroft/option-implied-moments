import numpy as np
import pytest

from src.ext.trapezoid_rnm import OPT_CALL, OPT_PUT, compute_trapz_rnm


def _make_indptr(sizes: list[int]) -> np.ndarray:
    """Build a CSR indptr array from a list of group sizes."""
    indptr = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    return indptr


def single_group_input(
    n_calls: int = 8,
    n_puts: int = 8,
    spot: float = 100.0,
    r: float = 0.02,
    ttm: float = 0.25,
    call_strikes: np.ndarray | None = None,
    put_strikes:  np.ndarray | None = None,
    ivol: float = 0.20,
) -> dict:
    """
    Build minimal valid single-group fixture for compute_trapz_rnm
    """
    # 105, 110, ... for calls; 95, 90, ... for puts (if not provided)
    if call_strikes is None:
        call_strikes = spot + np.arange(1, n_calls + 1) * 5.0
    if put_strikes is None:
        put_strikes  = spot - np.arange(1, n_puts  + 1) * 5.0

    strikes = np.concatenate([call_strikes, put_strikes])
    flags = np.array(
        [OPT_CALL] * len(call_strikes) + [OPT_PUT] * len(put_strikes),
        dtype=np.int32,
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
def _wide_grid_inputs(spot: float, ivol: float, ttm: float, n: int = 40) -> dict:  # noqa: E501
    """
    Build inputs with a strike grid that scales with implied volatility and TTM
    so the integration domain covers the tails regardless of the parameters.
    Uses +/-4 standard deviations in log-moneyness as the grid boundary.
    """
    std = ivol * np.sqrt(ttm)
    log_bounds = 4.0 * std

    call_strikes = spot * np.exp(np.linspace(0.01, log_bounds, n))
    put_strikes  = spot * np.exp(np.linspace(-log_bounds, -0.01, n))

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
        varQ, skewQ, kurtQ = compute_trapz_rnm(**valid_single_group_data)
        assert varQ.shape == (1,)
        assert skewQ.shape == (1,)
        assert kurtQ.shape == (1,)

    def test_return_types(self, valid_single_group_data):
        """
        Test that output arrays have correct dtypes for single group input.
        """
        varQ, skewQ, kurtQ = compute_trapz_rnm(**valid_single_group_data)
        assert varQ.dtype == np.float64
        assert skewQ.dtype == np.float64
        assert kurtQ.dtype == np.float64

    def test_variance_positive(self, valid_single_group_data):
        var, _, _ = compute_trapz_rnm(**valid_single_group_data)
        assert var[0] > 0.0

    def test_kurtosis_exceeds_one(self, valid_single_group_data):
        """
        For a symmetric IV smile, kurtosis should be well above 1.
        This is a loose sanity check, not a precise formula assertion.
        """
        _, _, kurt = compute_trapz_rnm(**valid_single_group_data)
        assert kurt[0] > 1.0

    def test_symmetric_smile_near_zero_skew(self):
        """
        A perfectly symmetric IV smile (equal put/call strikes equidistant
        from spot, equal IVols) should produce skewness close to zero.
        """
        spot = 100.0
        # 5,10,...,40
        offsets = np.arange(5, 45, 5, dtype=np.float64)
        inp = single_group_input(
            call_strikes=spot + offsets,
            put_strikes=spot  - offsets,
            ivol=0.20,
        )
        skew, _, _ = compute_trapz_rnm(**inp)
        # near zero, not exact due to BS nonlinearity
        assert abs(skew[0]) < 0.5

    def test_multiple_groups_independent(self):
        """
        Results for two groups must be independent of each other.
        """
        inp1 = single_group_input(ivol=0.20)
        inp2 = single_group_input(ivol=0.30)

        skew1, var1, _ = compute_trapz_rnm(**inp1)
        skew2, var2, _ = compute_trapz_rnm(**inp2)

        # Batch both groups in one call
        n1 = len(inp1["strikes"])
        n2 = len(inp2["strikes"])
        batch = dict(
            strikes=np.concatenate([inp1["strikes"], inp2["strikes"]]),
            ivols=np.concatenate([inp1["ivols"],   inp2["ivols"]]),
            flags=np.concatenate([inp1["flags"],   inp2["flags"]]),
            spots=np.array([100.0, 100.0]),
            rf=np.array([0.02, 0.02]),
            ttm=np.array([0.25, 0.25]),
            indptr=_make_indptr([n1, n2]),
        )
        bs, bv, _ = compute_trapz_rnm(**batch)

        np.testing.assert_allclose(bs[0], skew1[0], rtol=1e-10)
        np.testing.assert_allclose(bv[0], var1[0],  rtol=1e-10)
        np.testing.assert_allclose(bs[1], skew2[0], rtol=1e-10)
        np.testing.assert_allclose(bv[1], var2[0],  rtol=1e-10)

    def test_unsorted_strikes_give_same_result_as_sorted(self):
        """
        C layer must sort internally; hence input order must not affect output.
        """
        inp_sorted = single_group_input()

        inp_shuffled = {k: v.copy() for k, v in inp_sorted.items()}
        perm = (
            np.random.default_rng(42)
            .permutation(len(inp_sorted["strikes"]))
        )
        inp_shuffled["strikes"] = inp_sorted["strikes"][perm]
        inp_shuffled["ivols"] = inp_sorted["ivols"][perm]
        inp_shuffled["flags"] = inp_sorted["flags"][perm]

        s1, v1, k1 = compute_trapz_rnm(**inp_sorted)
        s2, v2, k2 = compute_trapz_rnm(**inp_shuffled)

        np.testing.assert_allclose(s1, s2, rtol=1e-10)
        np.testing.assert_allclose(v1, v2, rtol=1e-10)
        np.testing.assert_allclose(k1, k2, rtol=1e-10)

    def test_higher_ivol_higher_variance(self):
        """
        On a grid scaled to +/-4 sigma, variance must be strictly increasing
        with implied volatility.
        """
        spot  = 100.0
        ivols = [0.10, 0.20, 0.30, 0.40]
        vars_ = []
        for ivol in ivols:
            inp = _wide_grid_inputs(spot=spot, ivol=ivol, ttm=0.25)
            _, var, _ = compute_trapz_rnm(**inp)
            vars_.append(var[0])

        for i in range(len(vars_) - 1):
            assert vars_[i] < vars_[i + 1], (
                f"Variance not increasing: ivol={ivols[i]:.2f} -> var={vars_[i]:.6f}, "  # noqa: E501
                f"ivol={ivols[i+1]:.2f} -> var={vars_[i+1]:.6f}"
            )
