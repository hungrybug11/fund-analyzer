"""Data cleaning, alignment, and currency conversion."""

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_fx_warned = False


# ---------------------------------------------------------------------------
# Static FX rates (approximate, for long-term analysis)
# ---------------------------------------------------------------------------

FX_RATES: dict[str, dict[str, float]] = {
    # from_currency -> {to_currency: rate}
    "USD": {"CNY": 7.10, "HKD": 7.80},
    "CNY": {"USD": 1 / 7.10, "HKD": 1 / 0.91},
    "HKD": {"USD": 1 / 7.80, "CNY": 0.91},
}

# Naming: from_currency -> FX_RATES[from_currency][to_currency] gives
# how many units of to_currency per 1 unit of from_currency.


def normalize_dataframe(df: pd.DataFrame, fund_code: str) -> pd.DataFrame:
    """Ensure DataFrame has standard columns [date, nav] and is clean.

    Args:
        df: Raw DataFrame from fetcher.
        fund_code: Fund code for logging context.

    Returns:
        Clean DataFrame with columns [date, nav], sorted by date ascending.
    """
    if df is None or df.empty:
        raise ValueError(f"Empty DataFrame for {fund_code}")

    # Ensure we have date and nav columns
    if "date" not in df.columns or "nav" not in df.columns:
        raise ValueError(
            f"DataFrame for {fund_code} missing required columns. "
            f"Has: {list(df.columns)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df[df["nav"] > 0]  # NAV must be positive
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

    return df


def align_to_common_dates(
    navs: dict[str, pd.DataFrame],
    method: str = "inner",
) -> pd.DataFrame:
    """Align all fund NAV series to a common trading-date index.

    Default uses *inner* join (intersection of all dates). This ensures
    returns and risk metrics are computed over the same period for every
    fund — essential for valid comparison and optimization.

    Use ``method="outer"`` only for display charts where you want to see
    each fund's individual full history.

    Args:
        navs: Dict mapping fund_code -> DataFrame with columns [date, nav].
        method: "inner" (intersection, default) or "outer" (union).

    Returns:
        Wide-format DataFrame: index=date, columns=fund_code, values=NAV.
    """
    if not navs:
        raise ValueError("No NAV data to align.")

    aligned = None
    for code, df in navs.items():
        sub = df.set_index("date")[["nav"]].rename(columns={"nav": code})
        if aligned is None:
            aligned = sub
        else:
            aligned = aligned.join(sub, how=method)

    if aligned is None or len(aligned) < 20:
        raise ValueError(
            f"Only {len(aligned) if aligned is not None else 0} aligned dates. "
            f"Need at least 20 for meaningful analysis."
        )

    aligned = aligned.sort_index()

    if method == "inner":
        # No fill needed — all funds have data on every date
        return aligned

    # Outer join: forward-fill short gaps (weekends/holidays) but NOT
    # the long leading-NaN periods for funds that didn't exist yet
    aligned = aligned.ffill(limit=5)
    return aligned


def convert_currency(
    df: pd.DataFrame,
    from_currency: str,
    to_currency: str,
) -> pd.DataFrame:
    """Convert NAV values between currencies.

    Args:
        df: DataFrame with 'nav' column (or wide-format with fund columns).
        from_currency: Source currency code.
        to_currency: Target currency code.

    Returns:
        DataFrame with converted NAV values.
    """
    if from_currency == to_currency:
        return df

    global _fx_warned
    if not _fx_warned:
        logger.warning("Using static FX rates (USD/CNY=%.2f). Update FX_RATES in cleaner.py for accuracy.", FX_RATES["USD"]["CNY"])
        _fx_warned = True

    rate = FX_RATES.get(from_currency, {}).get(to_currency)
    if rate is None:
        raise ValueError(
            f"No FX rate found for {from_currency} -> {to_currency}. "
            f"Available rates: {list(FX_RATES.keys())}"
        )

    result = df.copy()
    if "nav" in result.columns:
        result["nav"] = result["nav"] * rate
    else:
        # Wide-format: convert all non-date columns
        for col in result.columns:
            if col != "date":
                result[col] = result[col] * rate

    return result
