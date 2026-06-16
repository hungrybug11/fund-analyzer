"""Local Parquet cache for fund NAV data.

Reads/writes NAV DataFrames to disk, tracks freshness.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


def cache_path(fund_code: str, data_dir: str | Path) -> Path:
    """Return the Parquet file path for a given fund code."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize fund code for filename (^ and . are common in Yahoo tickers)
    safe_code = fund_code.replace("^", "IDX_").replace(".", "_")
    return data_dir / f"{safe_code}.parquet"


def load_cache(fund_code: str, data_dir: str | Path) -> pd.DataFrame | None:
    """Load cached NAV DataFrame, or return None if missing or unreadable.

    Args:
        fund_code: Fund code / ticker.
        data_dir: Path to cache directory.

    Returns:
        DataFrame with columns [date, nav], or None if cache miss.
    """
    filepath = cache_path(fund_code, data_dir)
    if not filepath.exists():
        return None

    try:
        df = pd.read_parquet(filepath)
        if df.empty or "date" not in df.columns or "nav" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        # Corrupt cache file — delete and return None
        logger.warning("Corrupt cache deleted: %s (%s)", filepath, e)
        filepath.unlink(missing_ok=True)
        return None


def save_cache(fund_code: str, df: pd.DataFrame, data_dir: str | Path) -> None:
    """Save NAV DataFrame to local Parquet cache.

    Args:
        fund_code: Fund code / ticker.
        df: DataFrame with columns [date, nav].
        data_dir: Path to cache directory.
    """
    filepath = cache_path(fund_code, data_dir)
    df_to_save = df.copy()
    df_to_save["date"] = pd.to_datetime(df_to_save["date"])
    df_to_save.to_parquet(filepath, index=False)


def last_cached_date(fund_code: str, data_dir: str | Path) -> date | None:
    """Return the most recent date in cached data, or None."""
    df = load_cache(fund_code, data_dir)
    if df is None or df.empty:
        return None
    return df["date"].max()


def is_cache_valid(fund_code: str, data_dir: str | Path, max_age_days: int) -> bool:
    """Check if cache exists and is younger than max_age_days.

    Args:
        fund_code: Fund code / ticker.
        data_dir: Path to cache directory.
        max_age_days: Maximum allowed age in days.

    Returns:
        True if cache is fresh enough, False otherwise.
    """
    cached_date = last_cached_date(fund_code, data_dir)
    if cached_date is None:
        return False

    today = date.today()
    age = (today - cached_date).days
    return age <= max_age_days
