"""Portfolio optimization: efficient frontier, max Sharpe, risk parity, min variance.

All optimization uses scipy.optimize with long-only constraints (w>=0, sum(w)=1).
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def expected_returns(
    daily_returns: pd.DataFrame,
    method: str = "historical",
    periods_per_year: int = 252,
) -> np.ndarray:
    """Compute expected annualized returns for each fund.

    Args:
        daily_returns: DataFrame index=date, columns=fund_code.
        method: "historical" (mean).
        periods_per_year: Trading days per year.

    Returns:
        1D array of expected annualized returns, same order as columns.
    """
    if method == "historical":
        return daily_returns.mean().values * periods_per_year
    raise ValueError(f"Unknown method: {method}")


def covariance_matrix(
    daily_returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    periods_per_year: int = 252,
) -> np.ndarray:
    """Compute annualized covariance matrix.

    Args:
        daily_returns: DataFrame index=date, columns=fund_code.
        method: "sample" or "ledoit_wolf" (shrinkage).
        periods_per_year: Trading days per year.

    Returns:
        2D covariance matrix (n_assets 脳 n_assets), annualized.
    """
    if method == "sample":
        return daily_returns.cov().values * periods_per_year
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(daily_returns.dropna())
        return lw.covariance_ * periods_per_year
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Portfolio stat helpers
# ---------------------------------------------------------------------------

def portfolio_return(weights: np.ndarray, exp_ret: np.ndarray) -> float:
    """Annualized portfolio return."""
    return float(weights @ exp_ret)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualized portfolio volatility."""
    return float(np.sqrt(weights @ cov @ weights))


def portfolio_sharpe(
    weights: np.ndarray,
    exp_ret: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Portfolio Sharpe ratio."""
    pret = portfolio_return(weights, exp_ret)
    pvol = portfolio_volatility(weights, cov)
    if pvol < 1e-12:
        return 0.0
    return (pret - risk_free_rate) / pvol


# ---------------------------------------------------------------------------
# Optimization routines
# ---------------------------------------------------------------------------

def _make_bounds(n: int, max_weight: float = 0.40) -> list[tuple[float, float]]:
    """Long-only bounds: (0, max_weight) for each asset."""
    return [(0.0, max_weight)] * n


def _make_constraints(n: int) -> tuple[dict, ...]:
    """Constraints: sum(weights) = 1."""
    return (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )


def max_sharpe_portfolio(
    exp_ret: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    max_weight: float = 0.40,
) -> np.ndarray:
    """Find portfolio weights that maximize the Sharpe ratio.

    Args:
        exp_ret: Expected annualized returns (n_assets,).
        cov: Annualized covariance matrix (n_assets, n_assets).
        risk_free_rate: Risk-free rate.
        max_weight: Maximum allowed weight per asset.

    Returns:
        Optimal weight vector (n_assets,).
    """
    n = len(exp_ret)
    x0 = np.ones(n) / n  # equal weight start
    bounds = _make_bounds(n, max_weight)
    constraints = _make_constraints(n)

    def neg_sharpe(w):
        return -portfolio_sharpe(w, exp_ret, cov, risk_free_rate)

    result = minimize(
        neg_sharpe,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success:
        # Fall back to equal weight
        logger.warning("Max Sharpe optimization failed: %s. Falling back to equal weight.", result.message)
        return x0
    return result.x


def min_variance_portfolio(
    cov: np.ndarray,
    max_weight: float = 0.40,
) -> np.ndarray:
    """Find the minimum variance portfolio weights.

    Args:
        cov: Annualized covariance matrix.
        max_weight: Maximum allowed weight per asset.

    Returns:
        Optimal weight vector.
    """
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = _make_bounds(n, max_weight)
    constraints = _make_constraints(n)

    def port_vol(w):
        return portfolio_volatility(w, cov)

    result = minimize(
        port_vol,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning("Min variance optimization failed: %s. Falling back to equal weight.", result.message)
        return x0
    return result.x


def risk_parity_portfolio(
    cov: np.ndarray,
    max_weight: float = 0.40,
) -> np.ndarray:
    """Compute risk parity (equal risk contribution) weights.

    Minimizes: sum_i sum_j (RC_i - RC_j)^2
    where RC_i = w_i * (Cov @ w)_i / sqrt(w' Cov w).

    Args:
        cov: Annualized covariance matrix.
        max_weight: Maximum allowed weight per asset.

    Returns:
        Optimal weight vector.
    """
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = _make_bounds(n, max_weight)
    constraints = _make_constraints(n)

    def risk_parity_objective(w):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 1e12
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib / port_vol
        # Variance of risk contributions * n (equivalent to pairwise diff sum, O(n) vs O(n²))
        return float(np.var(risk_contrib) * n * n)

    result = minimize(
        risk_parity_objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning("Risk parity optimization failed: %s. Falling back to equal weight.", result.message)
        return x0
    return result.x


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier(
    exp_ret: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    num_points: int = 50,
    max_weight: float = 0.40,
) -> dict:
    """Compute the efficient frontier.

    Args:
        exp_ret: Expected annualized returns (n_assets,).
        cov: Annualized covariance matrix (n_assets, n_assets).
        risk_free_rate: Risk-free rate.
        num_points: Number of points on the frontier.
        max_weight: Maximum allowed weight per asset.

    Returns:
        dict with keys:
        - returns: list of frontier portfolio returns
        - volatilities: list of frontier portfolio volatilities
        - weights: list of weight arrays
        - sharpe_ratios: list of Sharpe ratios
    """
    # min_variance_portfolio is defined in this module (line ~144)

    n = len(exp_ret)

    # Find min variance portfolio as the leftmost point
    w_min = min_variance_portfolio(cov, max_weight)
    min_ret = portfolio_return(w_min, exp_ret)
    min_vol = portfolio_volatility(w_min, cov)

    # Max return = highest single-asset expected return
    max_ret = max(exp_ret)

    target_returns = np.linspace(min_ret, max_ret, num_points)
    bounds = _make_bounds(n, max_weight)

    frontier_returns = []
    frontier_vols = []
    frontier_weights = []
    frontier_sharpes = []

    for target in target_returns:
        x0 = np.ones(n) / n
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, exp_ret) - t},
        ]

        def port_vol_obj(w):
            return portfolio_volatility(w, cov)

        result = minimize(
            port_vol_obj,
            x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            w = result.x
            frontier_weights.append(w)
            frontier_returns.append(portfolio_return(w, exp_ret))
            frontier_vols.append(portfolio_volatility(w, cov))
            frontier_sharpes.append(portfolio_sharpe(w, exp_ret, cov, risk_free_rate))

    return {
        "returns": frontier_returns,
        "volatilities": frontier_vols,
        "weights": frontier_weights,
        "sharpe_ratios": frontier_sharpes,
    }
