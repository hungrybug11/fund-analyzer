"""Return calculations: daily, cumulative, annualized, rolling.

Works with wide-format NAV DataFrames: index=date, columns=fund_code.
"""

import numpy as np
import pandas as pd


def compute_returns(nav_df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute period returns from NAV prices.

    Args:
        nav_df: Index=date, columns=fund_code, values=NAV.
        method: "simple" (pct_change) or "log" (log returns).

    Returns:
        DataFrame of returns with same shape, first row is NaN.
    """
    if method == "simple":
        returns = nav_df.pct_change()
    elif method == "log":
        returns = np.log(nav_df / nav_df.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'.")
    return returns.dropna(how="all")


def compute_portfolio_returns(
    nav_df: pd.DataFrame,
    weights: dict[str, float],
    method: str = "simple",
) -> pd.Series:
    """Compute weighted portfolio daily returns.

    Args:
        nav_df: Wide-format NAV DataFrame (index=date, columns=fund_code).
        weights: Dict mapping fund_code -> target weight (should sum to 1).
        method: "simple" or "log".

    Returns:
        Series of portfolio daily returns, index=date.
    """
    returns = compute_returns(nav_df, method)
    # Align weights to columns present in returns
    available = [c for c in returns.columns if c in weights]
    if not available:
        raise ValueError("No matching fund codes between returns and weights.")
    w = pd.Series({c: weights[c] for c in available})
    w = w / w.sum()  # re-normalize
    port_returns = returns[available].dot(w[available])
    port_returns.name = "portfolio"
    return port_returns


def annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized return from a daily return series.

    Uses geometric linking: (1 + total_return)^(periods / n) - 1.
    """
    clean = daily_returns.dropna()
    if len(clean) == 0:
        return np.nan
    total_return = (1 + clean).prod()
    n = len(clean)
    return float(total_return ** (periods_per_year / n) - 1)


def cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    """Compute cumulative return series. Base = 1.0 (so 1.25 = +25%)."""
    return (1 + daily_returns.dropna()).cumprod()


def rolling_returns(
    daily_returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling annualized returns over a given window.

    Args:
        daily_returns: Daily return series.
        window: Rolling window in trading days (default 252 = ~1 year).
        periods_per_year: Trading days per year.

    Returns:
        Series of rolling annualized returns.
    """
    return daily_returns.rolling(window).apply(
        lambda x: (1 + x).prod() ** (periods_per_year / window) - 1,
        raw=False,
    )


def annualized_returns_table(
    daily_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute annualized return for each fund column.

    Returns:
        Series with fund_code as index and annualized return as value.
    """
    return daily_returns.apply(
        lambda col: annualized_return(col, periods_per_year)
    )
