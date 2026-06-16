"""Risk metrics: Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def annualized_volatility(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualized standard deviation of daily returns."""
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.025,
    periods_per_year: int = 252,
) -> float:
    """Sharpe ratio = (annualized_return - rf) / annualized_volatility."""
    from src.analysis.returns import annualized_return

    ann_ret = annualized_return(daily_returns, periods_per_year)
    vol = annualized_volatility(daily_returns, periods_per_year)
    if vol is None or np.isnan(vol) or vol == 0:
        return np.nan
    return (ann_ret - risk_free_rate) / vol


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.025,
    periods_per_year: int = 252,
) -> float:
    """Sortino ratio: uses downside deviation in denominator.

    Downside deviation = std of returns that are below 0 (or below rf).
    """
    from src.analysis.returns import annualized_return

    ann_ret = annualized_return(daily_returns, periods_per_year)
    clean = daily_returns.dropna()
    mar_daily = risk_free_rate / periods_per_year  # minimum acceptable return
    downside = clean[clean < mar_daily]
    if len(downside) < 2:
        return np.nan
    downside_std = downside.std() * np.sqrt(periods_per_year)
    if downside_std == 0:
        return np.nan
    return (ann_ret - risk_free_rate) / downside_std


def max_drawdown(
    daily_returns: pd.Series,
) -> dict:
    """Compute maximum drawdown and related info from daily returns.

    Returns dict with:
        - max_drawdown: float (negative, e.g. -0.25 means -25%)
        - peak_date: date of peak
        - trough_date: date of trough
        - recovery_date: date of recovery (or None if not recovered)
        - duration_days: peak-to-trough days
    """
    clean = daily_returns.dropna()
    if clean.empty:
        return {"max_drawdown": np.nan, "peak_date": None, "trough_date": None,
                "recovery_date": None, "duration_days": None}

    cum = (1 + clean).cumprod()
    cummax = cum.cummax()
    drawdown = (cum - cummax) / cummax

    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()

    if pd.isna(trough_idx):
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None,
                "recovery_date": None, "duration_days": 0}

    # Peak is the max point before the trough
    peak_idx = cum[:trough_idx].idxmax()

    # Recovery: first date after trough where cum >= previous peak
    peak_value = cummax[trough_idx]
    after_trough = cum[trough_idx:]
    recovery_mask = after_trough >= peak_value
    recovery_idx = None
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()

    duration = (trough_idx - peak_idx).days if peak_idx is not None else None

    return {
        "max_drawdown": round(float(max_dd), 4),
        "peak_date": str(peak_idx.date()) if hasattr(peak_idx, 'date') else str(peak_idx),
        "trough_date": str(trough_idx.date()) if hasattr(trough_idx, 'date') else str(trough_idx),
        "recovery_date": str(recovery_idx.date()) if recovery_idx and hasattr(recovery_idx, 'date') else None,
        "duration_days": duration,
    }


def calmar_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.025,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio = annualized_return / |max_drawdown|."""
    from src.analysis.returns import annualized_return

    ann_ret = annualized_return(daily_returns, periods_per_year)
    dd_info = max_drawdown(daily_returns)
    md = dd_info["max_drawdown"]
    if md is None or md == 0 or np.isnan(md):
        return np.nan
    return ann_ret / abs(md)


def value_at_risk(
    daily_returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Value at Risk at given confidence level.

    Args:
        daily_returns: Daily return series.
        confidence: Confidence level (0.95 = 95%).
        method: "historical" (percentile) or "parametric" (assumes normality).

    Returns:
        VaR as a positive number (e.g., 0.02 = 2% daily loss).
    """
    clean = daily_returns.dropna()
    if len(clean) < 10:
        return np.nan

    if method == "historical":
        var = -float(clean.quantile(1 - confidence))
    elif method == "parametric":
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        var = -(clean.mean() + z * clean.std())
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    return float(var)


def cvar(
    daily_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Conditional VaR (expected shortfall) — average loss beyond VaR.

    Returns:
        CVaR as a positive number.
    """
    clean = daily_returns.dropna()
    threshold = clean.quantile(1 - confidence)
    tail = clean[clean <= threshold]
    if len(tail) == 0:
        return np.nan
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def risk_metrics_table(
    daily_returns: pd.DataFrame,
    risk_free_rate: float = 0.025,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Generate a summary table of all risk/return metrics for each fund.

    Args:
        daily_returns: DataFrame index=date, columns=fund_code, values=daily returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading days per year.

    Returns:
        DataFrame: rows=metrics, columns=fund_codes.
    """
    from src.analysis.returns import annualized_return

    metrics = {}
    for col in daily_returns.columns:
        series = daily_returns[col].dropna()
        if len(series) < 20:
            continue

        ann_ret = annualized_return(series, periods_per_year)
        ann_vol = annualized_volatility(series, periods_per_year)
        dd_info = max_drawdown(series)
        md = dd_info.get("max_drawdown", np.nan)

        metrics[col] = {
            "年化收益率": ann_ret,
            "年化波动率": ann_vol,
            "夏普比率": sharpe_ratio(series, risk_free_rate, periods_per_year),
            "索提诺比率": sortino_ratio(series, risk_free_rate, periods_per_year),
            "卡玛比率": calmar_ratio(series, risk_free_rate, periods_per_year),
            "最大回撤": md if not np.isnan(md) else np.nan,
            "回撤持续(天)": dd_info.get("duration_days"),
            "VaR 95%": value_at_risk(series, 0.95),
            "CVaR 95%": cvar(series, 0.95),
        }

    return pd.DataFrame(metrics).T
