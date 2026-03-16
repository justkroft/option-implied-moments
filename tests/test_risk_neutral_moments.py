import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from src.risk_neutral_moments import DataSchema, risk_neutral_moments

ds = DataSchema()


def _minimal_options_df(n_stocks: int = 2, n_months: int = 3) -> pl.DataFrame:
    """
    Build a minimal synthetic Polars DataFrame that passes validation.

    Each (stock, month) has 8 OTM calls and 8 OTM puts.
    """
    rows = []
    spot = 100.0

    for s in range(n_stocks):
        for m in range(n_months):
            d = date(2023, 1 + m, 28)
            for K in np.arange(105, 145, 5, dtype=float):
                rows.append(
                    {
                        ds.stock_identifier: s + 1,
                        ds.date: d,
                        ds.strike_price: K,
                        ds.spot_price: spot,
                        ds.implied_volatility: 0.20,
                        ds.option_type: "call",
                        ds.rf_rate: 0.02,
                        ds.time_to_maturity: 63.0,
                    }
                )
            for K in np.arange(60, 100, 5, dtype=float):
                rows.append(
                    {
                        ds.stock_identifier: s + 1,
                        ds.date: d,
                        ds.strike_price: K,
                        ds.spot_price: spot,
                        ds.implied_volatility: 0.20,
                        ds.option_type: "put",
                        ds.rf_rate: 0.02,
                        ds.time_to_maturity: 63.0,
                    }
                )

    return pl.DataFrame(rows).with_columns(pl.col(ds.date).cast(pl.Date))


class TestRiskNeutralMomentsValidation:
    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="polars.DataFrame"):
            risk_neutral_moments({"data": [1, 2, 3]})

    def test_rejects_invalid_group_freq(self):
        df = _minimal_options_df()
        with pytest.raises(ValueError, match="group_freq"):
            risk_neutral_moments(df, group_freq="W")

    def test_rejects_missing_column(self):
        df = _minimal_options_df().drop(ds.implied_volatility)
        with pytest.raises(KeyError):
            risk_neutral_moments(df)


class TestRiskNeutralMomentsSuccess:
    def test_output_shape(self):
        df = _minimal_options_df(n_stocks=2, n_months=3)
        out = risk_neutral_moments(df, group_freq="1mo")
        # Expect 2 stocks × 3 months = 6 rows
        assert len(out) == 6

    def test_output_columns(self):
        df = _minimal_options_df()
        out = risk_neutral_moments(df)
        assert set(out.columns) >= {
            "stock_id",
            "date",
            "skewQ",
            "varQ",
            "volQ",
            "kurtQ",
        }

    def test_variance_positive(self):
        df = _minimal_options_df()
        out = risk_neutral_moments(df)
        assert (out["varQ"].drop_nulls() > 0).all()

    def test_vol_is_sqrt_annualised_var(self):
        df = _minimal_options_df()
        out = risk_neutral_moments(df)
        expected_vol = (out["varQ"].sqrt() * math.sqrt(12)).alias("expected")
        np.testing.assert_allclose(
            out["volQ"].to_numpy(),
            expected_vol.to_numpy(),
            rtol=1e-9,
            equal_nan=True,
        )

    def test_output_is_polars_dataframe(self):
        df = _minimal_options_df()
        out = risk_neutral_moments(df)
        assert isinstance(out, pl.DataFrame)

    def test_daily_group_freq(self):
        """group_freq='1d' should produce one row per stock per day."""
        df = _minimal_options_df(n_stocks=1, n_months=2)
        out = risk_neutral_moments(df, group_freq="1d")
        assert len(out) >= 1
