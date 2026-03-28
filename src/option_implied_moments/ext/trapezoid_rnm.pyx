from cython.parallel cimport prange
from libc.stdint cimport int32_t

import numpy as np
from numpy cimport intp_t


# C declarations
cdef extern from "trapezoid_core.h" nogil:

    int TRAPZ_OK
    int TRAPZ_ERR_ALLOC
    int TRAPZ_ERR_NO_CALLS
    int TRAPZ_ERR_NO_PUTS
    int TRAPZ_ERR_FEW_OPT

    int trapz_moments(
        size_t n,
        const double *strikes,
        const double *ivols,
        const int *flags,
        double spot,
        double r,
        double T,
        double *out_var,
        double *out_skew,
        double *out_kurt
    )

# Same as OPT_CALL / OPT_PUT in trapezoid_core.h. Mind to keep in sync
OPT_CALL = 1
OPT_PUT  = 0


def compute_trapz_rnm(
    double[::1] strikes,
    double[::1] ivols,
    int[::1] flags,
    double[::1] spots,
    double[::1] rf,
    double[::1] ttm,
    intp_t[::1] indptr,
):
    """
    Compute risk-neutral moments for all groups in parallel.

    Bridge between the Python API and the C helper functions. This function does the
    following:
    - Accept pre-processed flat NumPy arrays + CSR-style group index pointers.
    - Validate input consistency and data availability.
    - Iterate over groups in parallel with OpenMP prange.
    - Call C-code under nogil for each group.
    - Write scalar results into pre-allocated output arrays; return as NumPy arrays

    Parameters
    ----------
    strikes : ndarray, shape (N,)
        OTM strike prices across all groups.
    ivols : ndarray, shape (N,)
        Corresponding implied volatilities.
    flags : ndarray, shape (N,), dtype int
        Option type: OPT_CALL (1) or OPT_PUT (0).
    spots : ndarray, shape (G,)
        Spot price for each group.
    rf : ndarray, shape (G,)
        Risk-free rate for each group.
    ttm : ndarray, shape (G,)
        Time-to-maturity in years for each group.
    indptr : ndarray, shape (G+1,), dtype intp_t
        CSR-style row pointers.  Group i owns the slice
        [indptr[i], indptr[i+1]) of the flat arrays.

    Returns
    -------
    var : ndarray, shape (G,), float64
    skew : ndarray, shape (G,), float64
    kurt : ndarray, shape (G,), float64
    rc : ndarray, shape (G,), int32
        Risk-neutral moments per group and return code; NaN where computation failed.

    Raises
    ------
    ValueError
        If strikes, ivols, or flags have inconsistent lengths, if array lengths
        don't match the CSR structure, or if any group has fewer than 4 options.
    """

    cdef:
        intp_t n_groups = indptr.shape[0] - 1
        intp_t g, start, end, seg_len

        # Convert to contiguous typed memoryviews
        double[::1] mv_strikes = np.ascontiguousarray(strikes)
        double[::1] mv_ivols = np.ascontiguousarray(ivols)
        int[::1] mv_flags = np.ascontiguousarray(flags)
        double[::1] mv_spots = np.ascontiguousarray(spots)
        double[::1] mv_rf = np.ascontiguousarray(rf)
        double[::1] mv_ttm = np.ascontiguousarray(ttm)
        intp_t[::1] mv_indptr = np.ascontiguousarray(indptr)

        # output arrays - initialize to NaN/0 to avoid uninitialized memory
        double[::1] out_var = np.full(n_groups, np.nan, dtype=np.float64)
        double[::1] out_skew = np.full(n_groups, np.nan, dtype=np.float64)
        double[::1] out_kurt = np.full(n_groups, np.nan, dtype=np.float64)
        int[::1] out_rc = np.zeros(n_groups, dtype=np.int32)

        int rc

    # input validation
    if len(strikes) != len(ivols) or len(strikes) != len(flags):
        raise ValueError(
            f"strikes ({len(strikes)}), ivols ({len(ivols)}), and flags "
            f"({len(flags)}) must all have the same length"
        )

    if len(strikes) == 0:
        raise ValueError("No options provided (strikes array is empty)")

    if len(spots) != n_groups or len(rf) != n_groups or len(ttm) != n_groups:
        raise ValueError(
            f"spots, rf, and ttm must have length equal to number of groups "
            f"({n_groups}), got {len(spots)}, {len(rf)}, {len(ttm)}"
        )

    if indptr[0] != 0:
        raise ValueError("indptr must start at 0")

    if indptr[n_groups] != len(strikes):
        raise ValueError(
            f"indptr final value ({indptr[n_groups]}) must equal "
            f"number of strikes ({len(strikes)})"
        )

    # Check that each group has at least 4 options
    for g in range(n_groups):
        seg_len = indptr[g + 1] - indptr[g]
        if seg_len < 4:
            raise ValueError(
                f"Group {g} has only {seg_len} option(s), "
                f"but at least 4 are required for moment computation"
            )

    for g in prange(n_groups, nogil=True, schedule="dynamic"):
        start = mv_indptr[g]
        end = mv_indptr[g + 1]
        seg_len = end - start

        rc = trapz_moments(
            <size_t>seg_len,
            &mv_strikes[start],
            &mv_ivols[start],
            &mv_flags[start],
            mv_spots[g],
            mv_rf[g],
            mv_ttm[g],
            &out_var[g],
            &out_skew[g],
            &out_kurt[g]
        )
        out_rc[g] = rc

    return (
        np.asarray(out_var),
        np.asarray(out_skew),
        np.asarray(out_kurt),
        np.asarray(out_rc)
    )
