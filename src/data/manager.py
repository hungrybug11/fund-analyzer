"""Data pipeline orchestrator.

Coordinates cache checks, data fetching, cleaning, and alignment.
Returns ready-to-analyze DataFrames.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from src.config import FundConfig, PortfolioConfig

from .cache import is_cache_valid, load_cache, save_cache
from .cleaner import align_to_common_dates, convert_currency, normalize_dataframe
from .fetcher import DataFetchError, fetch_fund, fetch_with_retry

logger = logging.getLogger(__name__)


class FetchResult:
    """Result of a data pipeline run."""

    def __init__(self):
        self.successes: dict[str, pd.DataFrame] = {}
        self.failures: dict[str, str] = {}  # code -> error message
        self.warnings: list[str] = []

    @property
    def has_data(self) -> bool:
        return len(self.successes) > 0



def get_fund_data(fund: FundConfig, config: PortfolioConfig) -> Optional[pd.DataFrame]:
    """Get cleaned NAV series for a single fund (cache-aware).

    1. Check cache. If valid, return cached data.
    2. Otherwise, fetch from source, save to cache, return.

    Args:
        fund: Fund configuration.
        config: Full portfolio configuration.

    Returns:
        DataFrame with columns [date, nav], or None if fetch fails.
    """
    data_dir = config.output.data_dir
    cache_expiry = config.output.cache_expiry_days

    # Check cache first
    if is_cache_valid(fund.code, data_dir, cache_expiry):
        df = load_cache(fund.code, data_dir)
        if df is not None:
            logger.info("Using cached data for %s (%s)", fund.code, fund.name)
            return df

    # Fetch from source
    today = date.today()
    start_date = (today - timedelta(days=config.analysis.lookback_years * 365)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    logger.info("Fetching %s (%s) from source...", fund.code, fund.name)

    try:
        df = fetch_with_retry(
            fetch_fund,
            fund.type,
            fund.code,
            start_date,
            end_date,
            fund_label=f"{fund.code} ({fund.name})",
        )

        # Clean
        df = normalize_dataframe(df, fund.code)

        # Currency conversion
        if fund.currency != config.base_currency:
            df = convert_currency(df, fund.currency, config.base_currency)

        # Cache
        save_cache(fund.code, df, data_dir)
        logger.info("Cached data for %s (%d rows)", fund.code, len(df))

        return df

    except DataFetchError as e:
        logger.warning("Fetch failed for %s: %s", fund.code, e)
        # Try stale cache as fallback
        stale = load_cache(fund.code, data_dir)
        if stale is not None:
            logger.info("Using stale cache for %s", fund.code)
            return stale
        return None
    except Exception as e:
        logger.error("Unexpected error fetching %s: %s", fund.code, e)
        stale = load_cache(fund.code, data_dir)
        if stale is not None:
            return stale
        return None


def get_portfolio_data(
    config: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Main data pipeline entry point.

    1. For each fund: check cache -> fetch if needed -> clean -> convert currency.
    2. Align all funds to common date index (inner for analysis, outer for display).

    Returns:
        (portfolio_navs, navs_display, warnings)
    """
    result = FetchResult()

    # Step 1: Fetch all funds
    fund_navs = {}
    n_funds = len(config.funds)
    for i, fund in enumerate(config.funds, 1):
        logger.info("Fetching %d/%d: %s (%s)", i, n_funds, fund.code, fund.name)
        df = get_fund_data(fund, config)
        if df is not None and not df.empty:
            fund_navs[fund.code] = df
            result.successes[fund.code] = df
        else:
            result.failures[fund.code] = f"Failed to fetch data for {fund.code} ({fund.name})"
            result.warnings.append(f"Skipping {fund.code} ({fund.name}): no data available")

    if not fund_navs:
        raise RuntimeError(
            "Failed to fetch data for all funds. Check your config and network connection."
        )

    # Step 2: Align — inner for analysis, outer for display
    try:
        portfolio_navs = align_to_common_dates(fund_navs, method="inner")
        navs_display = align_to_common_dates(fund_navs, method="outer")
    except ValueError as e:
        raise RuntimeError(f"Data alignment failed: {e}")

    # Report
    logger.info(
        "Portfolio data ready: %d funds, %d dates",
        len(portfolio_navs.columns),
        len(portfolio_navs),
    )
    if result.warnings:
        for w in result.warnings:
            logger.warning(w)

    return portfolio_navs, navs_display, result.warnings
