import numpy as np
import polars as pl

from src.ext.trapezoid_rnm import OPT_CALL, OPT_PUT, compute_trapz_rnm

REQUIRED_COLUMNS = [
    "stock_identifier",
    "date",
    "option_type",
    "spot_price",
    "strike_price",
    "time_to_maturity",
    "implied_volatility",
    "rf_rate"
]


def risk_neutral_moments(
    options_data: pl.DataFrame,
    group_freq: str = "1mo"
) -> pl.DataFrame:

    # Filter OTM options
    # Call OTM: Strike > spot price, Put OTM: Strike < spot price
    otm = (
        options_data
        .filter(
            pl.when(pl.col("option_type") == "call")
            .then(pl.col("strike_price") > pl.col("spot_price"))
            .otherwise(pl.col("strike_price") < pl.col("spot_price"))
        )
        # encode flag as integer
        .with_columns(
            pl.when(pl.col("option_type") == "call")
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
        .sort(["stock_identifier", "date"])
        .with_columns(pl.col("date").dt.truncate(group_freq).alias("_period"))
        .sort(["stock_identifier", "_period"])  # resort
        .group_by(["stock_identifier", "_period"], maintain_order=True)
        .agg([
            pl.col("spot_price").first().alias("_spot"),
            pl.col("rf_rate").first().alias("_r"),
            (pl.col("time_to_maturity") / 252.0).first().alias("_T"),
            pl.len().alias("_n_opts"),
        ])
        .sort(["stock_identifier", "_period"])
    )

    n_groups: int = len(groups)

    # Build indptr from group sizes
    group_sizes = groups["_n_opts"].to_numpy().astype(np.int64)
    indptr = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(group_sizes, out=indptr[1:])

    # We need the flat option arrays ordered consistently with `groups`.
    # Re-sort otm to match the groups sort order, then extract arrays.
    # The sort above (stock_id, _period) already gives us this order.
    flat_strikes = otm["strike_price"].to_numpy().astype(np.float64)
    flat_ivols = otm["implied_volatility"].to_numpy().astype(np.float64)
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
            pl.col("stock_identifier").alias("stock_id"),
            pl.col("_period").alias("date"),
        ).with_columns([
            pl.Series("varQ", varQ, dtype=pl.Float64),
            pl.Series("skewQ", skewQ, dtype=pl.Float64),
            pl.Series(
                "volQ",
                np.sqrt(np.where(varQ > 0, varQ, np.nan)) * np.sqrt(12.0),
                dtype=pl.Float64
            ),
            pl.Series("kurtQ", kurtQ,  dtype=pl.Float64),
        ])
    )
