from cython.parallel cimport prange

import numpy as np

# type declarations
ctypedef Py_ssize_t intp_t
ctypedef double float64_t
ctypedef int int32_t
ctypedef long int64_t

# C declarations
cdef extern from "trapezoid_core.h" nogil:

    int TRAPZ_OK
    int TRAPZ_ERR_ALLOC
    int TRAPZ_ERR_NO_CALLS
    int TRAPZ_ERR_NO_PUTS
    int TRAPZ_ERR_FEW_OPT

    int OPT_CALL
    int OPT_PUT

    ctypedef struct TrapezoidResult:
        double skew
        double var
        double kurt

    int trapz_moments(
        size_t        n,
        const double *strikes,
        const double *ivols,
        const int    *flags,
        double        spot,
        double        r,
        double        T,
        RNMResult    *out
    )


def compute_trapz_rnm(
    float64_t[::1] strikes,
    float64_t[::1] ivols,
    int32_t[::1] flags,
    float64_t[::1] spots,
    float64_t[::1] rf,
    float64_t[::1] ttm,
    int64_t[::1] indptr,
):

    cdef:
        intp_t n_groups = indptr.shape[0] - 1
        intp_t g, start, end, seg_len

        # contiguous typed memory views
        # TODO: check if this is necessary, or if we can just use the input
        # arrays directly
        # float64_t[::1] mv_strikes = np.ascontiguousarray(strikes)
        # float64_t[::1] mv_ivols   = np.ascontiguousarray(ivols)
        # int32_t[::1] mv_flags     = np.ascontiguousarray(flags)
        # float64_t[::1] mv_spots    = np.ascontiguousarray(spots)
        # float64_t[::1] mv_rf       = np.ascontiguousarray(rf)
        # float64_t[::1] mv_ttm      = np.ascontiguousarray(ttm)
        # int64_t[::1] mv_indptr    = np.ascontiguousarray(indptr)

        # pre-allocate output arrays
        float64_t[::1] out_var = np.empty(n_groups, dtype=np.float64)
        float64_t[::1] out_skew = np.empty(n_groups, dtype=np.float64)
        float64_t[::1] out_kurt = np.empty(n_groups, dtype=np.float64)

        TrapezoidResult result
        int rc
