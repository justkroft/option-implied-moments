import numpy as np
from src.ext.trapezoid_rnm import OPT_CALL, OPT_PUT


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
    """Build a single-group fixture for compute_trapz_rnm."""
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
