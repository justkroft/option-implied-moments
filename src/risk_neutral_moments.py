from dataclasses import dataclass

import numpy as np
import polars as pl

from src.ext.trapezoid_rnm import OPT_CALL, OPT_PUT, compute_trapz_rnm


@dataclass(frozen=True)
class DataSchema:
    """
    A class to hold the column names for the input data

    Parameters
    ----------
    stock_identifier : str
        Key identifier for the stocks, e.g., the CRSP PERMNO.
        By default, "stock_id" is used as the column name for this field.
    date : str
        The date of the observation. By default, "date" is used as the column
        name for this field.
    option_type : str
        Call / Put flag identifier column name. By default, "option_type" is
        used as the column name for this. The expectation of values in this
        column can be adjusted with the `call_flag` and `put_flag` parameters.
    call_flag : str
        The value in the `option_type` column that indicates a call option. By
        default, "call" is used as the call flag.
    put_flag : str
        The value in the `option_type` column that indicates a put option. By
        default, "put" is used as the put flag.
    spot_price : str
        The price of the underlying; stock price. By default, "spot_price" is
        used as the column name for this field.
    strike_price : str
        The strike price of the underlying. By default, "strike_price" is
        used as the column name for this field.
    time_to_maturity : str
        The time-to-maturity in days. By default, "time_to_maturity" is
        used as the column name for this field.
    implied_volatility : str
        Implied volatility. By default, "implied_volatility" is
        used as the column name for this field.
    rf_rate : str
        The risk-free rate. By default, "rf_rate" is
        used as the column name for this field.
    """
    stock_identifier: str = "stock_id"
    date: str = "date"
    option_type: str = "option_type"
    call_flag: str = "call"
    put_flag: str = "put"
    spot_price: str = "spot_price"
    strike_price: str = "strike_price"
    time_to_maturity: str = "time_to_maturity"
    implied_volatility: str = "implied_volatility"
    rf_rate: str = "rf_rate"


def risk_neutral_moments(
    options_data: pl.DataFrame,
    group_freq: str = "1mo",
    data_schema: DataSchema = DataSchema(),
) -> pl.DataFrame:
    """
    Compute risk-neutral variance, skewness, and kurtosis for every
    stock x period group in `options_data` using the trapezoidal
    integral approach of [2]_. This function uses a trapezoidal integral
    approach to estimate the integrals of the volatility, cubic, and quartic
    contracts of [1]_. See Appendix B2 and B3. of [2]_.

    This is a convenience wrapper around the core Cython function
    `compute_trapz_rnm` that handles the data manipulation and grouping logic.
    The input dataframe is expected to be in "long" format.

    Parameters
    ----------
    options_data : pl.DataFrame
        Must contain the columns defined in the `data_schema` parameter..
        Expected to hold OTM call and put options only (the caller is
        responsible for pre-filtering).
    group_freq : str
        Polars duration string used for the time grouper, e.g. ``"1mo"``
        (monthly, default) or ``"1d"`` (daily).
    data_schema : DataSchema, optional
        The data schema that holds the expected column names, by default
        DataSchema().

    Returns
    -------
    pl.DataFrame
        One row per (stock_identifier, period) with columns:
        ``stock_id``, ``date``, ``skewQ``, ``varQ``, ``volQ``, ``kurtQ``.

    Raises
    ------
    TypeError
        If `options_data` is not a Polars DataFrame.
    ValueError
        If `group_freq` is not ``"1mo"`` or ``"1d"``.
    KeyError
        If a required column is absent from `options_data`.

    References
    ----------
    .. [1] Bakshi, G., Kapadia, N., & Madan, D. (2003). Stock return
           characteristics, skew laws, and the differential pricing of
           individual equity options. The Review of Financial Studies, 16(1),
           101-143.
    .. [2] Bali, T. G., Hu, J., & Murray, S. (2019). Option implied volatility,
           skewness, and kurtosis and the cross-section of expected stock
           returns. Georgetown McDonough School of Business Research Paper.
    """
    if not isinstance(options_data, pl.DataFrame):
        raise TypeError(
            "options_data must be a polars.DataFrame, "
            f"got {type(options_data).__name__}."
        )

    valid_freqs = {"1mo", "1d"}
    if group_freq not in valid_freqs:
        raise ValueError(
            f"group_freq must be one of {valid_freqs}, got {group_freq!r}."
        )

    ds = data_schema  # alias for brevity
    required_cols = {
        ds.stock_identifier, ds.date, ds.option_type, ds.spot_price,
        ds.strike_price, ds.time_to_maturity, ds.implied_volatility, ds.rf_rate
    }
    missing = required_cols - set(options_data.columns)
    if missing:
        raise KeyError(f"options_data is missing required columns: {missing}")

    # Filter OTM options
    # Call OTM: Strike > spot price, Put OTM: Strike < spot price
    otm = (
        options_data
        .filter(
            pl.when(pl.col(ds.option_type) == ds.call_flag)
            .then(pl.col(ds.strike_price) > pl.col(ds.spot_price))
            .otherwise(pl.col(ds.strike_price) < pl.col(ds.spot_price))
        )
        # encode flag as integer
        .with_columns(
            pl.when(pl.col(ds.option_type) == ds.call_flag)
            .then(pl.lit(OPT_CALL, dtype=pl.Int32))
            .otherwise(pl.lit(OPT_PUT, dtype=pl.Int32))
            .alias("_flag_int")
        )
    )

    # Group by stock identifier and date; build CSR layout
    # We want one row in the groups table per (stock, period) pair, and flat
    # arrays for the option-level data, concatenated in the same order
    groups = (
        otm
        .with_columns(pl.col(ds.date).dt.truncate(group_freq).alias("_period"))
        .sort([ds.stock_identifier, "_period"])
        .group_by([ds.stock_identifier, "_period"], maintain_order=True)
        .agg([
            pl.col(ds.spot_price).first().alias("_spot"),
            pl.col(ds.rf_rate).first().alias("_r"),
            (pl.col(ds.time_to_maturity) / 252.0).first().alias("_T"),
            pl.len().alias("_n_opts"),
        ])
    )

    n_groups: int = len(groups)

    # Build indptr from group sizes
    group_sizes = groups["_n_opts"].to_numpy().astype(np.int64)
    indptr = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(group_sizes, out=indptr[1:])

    # We need the flat option arrays ordered consistently with `groups`.
    # sort otm to match the groups sort order, then extract arrays
    # The sort above (stock_id, _period) gives this order.
    flat_strikes = otm[ds.strike_price].to_numpy().astype(np.float64)
    flat_ivols = otm[ds.implied_volatility].to_numpy().astype(np.float64)
    flat_flags = otm["_flag_int"].to_numpy().astype(np.int32)

    per_group_spot = groups["_spot"].to_numpy().astype(np.float64)
    per_group_rf = groups["_r"].to_numpy().astype(np.float64)
    per_group_ttm = groups["_T"].to_numpy().astype(np.float64)

    # dispatch to Cython function
    varQ, skewQ, kurtQ = compute_trapz_rnm(
        flat_strikes,
        flat_ivols,
        flat_flags,
        per_group_spot,
        per_group_rf,
        per_group_ttm,
        indptr,
    )

    # reconstruct output dataframe
    return (
        groups
        .select(
            pl.col(ds.stock_identifier).alias("stock_id"),
            pl.col("_period").alias("date"),
        ).with_columns([
            pl.Series("varQ", varQ, dtype=pl.Float64),
            pl.Series(
                "volQ",
                np.sqrt(np.where(varQ > 0, varQ, np.nan)) * np.sqrt(12.0),
                dtype=pl.Float64
            ),
            pl.Series("skewQ", skewQ, dtype=pl.Float64),
            pl.Series("kurtQ", kurtQ,  dtype=pl.Float64),
        ])
    )
