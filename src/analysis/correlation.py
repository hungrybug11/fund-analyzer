"""Correlation analysis: Pearson matrix, distance matrix, hierarchical clustering."""

import numpy as np
import pandas as pd


def correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily returns.

    Args:
        daily_returns: DataFrame index=date, columns=fund_code.

    Returns:
        Square DataFrame of correlation coefficients.
    """
    return daily_returns.corr()


def distance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation to distance matrix: sqrt(2 * (1 - corr)).

    Distance is 0 when corr=1, ~2 when corr=-1.
    """
    corr = correlation_matrix(daily_returns)
    return np.sqrt(np.maximum(2 * (1 - corr), 0))


def hierarchical_clustering(
    daily_returns: pd.DataFrame,
    method: str = "ward",
):
    """Perform hierarchical clustering on the correlation distance matrix.

    Args:
        daily_returns: DataFrame index=date, columns=fund_code.
        method: Linkage method ("ward", "average", "complete", "single").

    Returns:
        (linkage_matrix, labels) where:
        - linkage_matrix: scipy linkage result
        - labels: fund codes in original order
    """
    from scipy.cluster.hierarchy import linkage

    dist = distance_matrix(daily_returns)
    # Convert condensed distance matrix (upper triangle)
    condensed = _square_to_condensed(dist.values)
    Z = linkage(condensed, method=method)
    return Z, list(dist.columns)


def _square_to_condensed(square: np.ndarray) -> np.ndarray:
    """Convert square distance matrix to condensed 1D form used by scipy."""
    n = square.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(square[i, j])
    return np.array(condensed)


def rolling_correlation(
    daily_returns: pd.DataFrame,
    fund_a: str,
    fund_b: str,
    window: int = 60,
) -> pd.Series:
    """Rolling Pearson correlation between two funds.

    Args:
        daily_returns: DataFrame index=date, columns=fund_code.
        fund_a: First fund code.
        fund_b: Second fund code.
        window: Rolling window in trading days.

    Returns:
        Series of rolling correlations, index=date.
    """
    if fund_a not in daily_returns.columns or fund_b not in daily_returns.columns:
        raise ValueError(f"Funds {fund_a} or {fund_b} not in returns DataFrame.")
    return daily_returns[fund_a].rolling(window).corr(daily_returns[fund_b])


def effective_n(corr_matrix: pd.DataFrame) -> float:
    """Compute effective number of independent bets.

    Based on correlation matrix: N_eff = N / (1 + (N-1)*avg_corr).
    Lower N_eff means more concentration / less diversification.
    """
    n = len(corr_matrix)
    if n <= 1:
        return n

    # Average off-diagonal correlation
    values = corr_matrix.values.copy()
    np.fill_diagonal(values, np.nan)
    avg_corr = np.nanmean(values)

    if avg_corr >= 1.0:
        return 1  # perfectly correlated

    neff = n / (1 + (n - 1) * avg_corr)
    return round(neff, 1)
