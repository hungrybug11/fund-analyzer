"""Data fetcher: all data via EastMoney direct APIs (same as cn-funds-mcp).

No akshare dependency. Uses urllib from stdlib only.
All fetch functions return standardized DataFrame with [date, nav] columns.
"""

import json
import logging
import re
import time
import urllib.request
from datetime import date, datetime
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when data fetch fails after all retries."""
    pass


BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fund.eastmoney.com/",
}

# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def fetch_with_retry(
    fetch_fn: Callable,
    *args,
    max_retries: int = 3,
    fund_label: str = "",
    **kwargs,
) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            df = fetch_fn(*args, **kwargs)
            if df is None or df.empty:
                raise DataFetchError(f"Empty response for {fund_label}")
            return df
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Fetch attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    attempt, max_retries, fund_label, e, wait,
                )
                time.sleep(wait)

    raise DataFetchError(
        f"Failed to fetch data for {fund_label} after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=BASE_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers=BASE_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ======================================================================
# A-share mutual funds -- F10DataApi (same source as cn-funds-mcp)
# ======================================================================

def fetch_a_share_mf(fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch A-share mutual fund NAV history via EastMoney F10DataApi.

    Same data source as cn-funds-mcp. Paginates through 20-row pages.
    """
    sdate = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    edate = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    # First request to get total records/pages
    url1 = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page=1&per=20&sdate={sdate}&edate={edate}"
    text = _get_text(url1)

    records_m = re.search(r'records:(\d+)', text)
    pages_m = re.search(r'pages:(\d+)', text)
    if not records_m:
        raise DataFetchError(f"Cannot parse records for {fund_code}")

    total_pages = int(pages_m.group(1)) if pages_m else 1

    # Parse all pages
    all_rows = []
    for page in range(1, total_pages + 1):
        if page > 1:
            url = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page}&per=20&sdate={sdate}&edate={edate}"
            text = _get_text(url)

        rows = re.findall(
            r'<tr><td>(\d{4}-\d{2}-\d{2})</td><td[^>]*>([^<]+)</td><td[^>]*>([^<]+)</td>',
            text
        )
        all_rows.extend(rows)

    if not all_rows:
        raise DataFetchError(f"No NAV data for {fund_code}")

    df = pd.DataFrame(all_rows, columns=["date", "nav", "total_nav"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date")

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    logger.info("Fetched %d NAV rows for %s (%s ~ %s)", len(df), fund_code, df["date"].iloc[0], df["date"].iloc[-1])

    return df[["date", "nav"]].reset_index(drop=True)


def fetch_a_share_etf(etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch A-share ETF NAV -- same API as mutual funds."""
    return fetch_a_share_mf(etf_code, start_date, end_date)


# ======================================================================
# A-share market indices -- push2 API
# ======================================================================

def fetch_akshare_index(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch A-share index data via EastMoney push2 API."""
    if index_code.startswith("000") or index_code.startswith("60"):
        secid = f"1.{index_code}"
    else:
        secid = f"0.{index_code}"

    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&beg={start_date}&end={end_date}&lmt=5000"
    data = _get_json(url)

    if not data.get("data") or not data["data"].get("klines"):
        raise DataFetchError(f"No index data for {index_code}")

    rows = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        rows.append({"date": parts[0], "value": float(parts[2])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    return df[["date", "value"]].reset_index(drop=True)


# ======================================================================
# US-listed ETFs -- overseas API
# ======================================================================

def fetch_overseas_etf(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch US-listed ETF via EastMoney overseas API."""
    ticker_upper = ticker.upper()

    for market in ["105", "106", "107"]:
        secid = f"{market}.{ticker_upper}"
        url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&beg={start_date}&end={end_date}&lmt=5000"
        try:
            data = _get_json(url)
            if data.get("data") and data["data"].get("klines"):
                break
        except Exception:
            continue
    else:
        raise DataFetchError(f"No data for US ETF {ticker}")

    rows = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        rows.append({"date": parts[0], "nav": float(parts[2])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date")

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    return df[["date", "nav"]].reset_index(drop=True)


# ======================================================================
# Global market indices
# ======================================================================

_GLOBAL_INDEX_MAP: dict[str, dict] = {
    "^GSPC": {"secid": "100.INX"},
    "^IXIC": {"secid": "100.IXIC"},
    "^DJI":  {"secid": "100.DJI"},
    "^RUT":  {"secid": "100.RUT"},
    "^N225": {"secid": "100.N225"},
    "^HSI":  {"secid": "100.HSI"},
}


def fetch_global_index(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    mapping = _GLOBAL_INDEX_MAP.get(index_code)
    if mapping is None:
        raise DataFetchError(f"Unknown global index: {index_code}")

    secid = mapping["secid"]
    url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&beg={start_date}&end={end_date}&lmt=5000"
    data = _get_json(url)

    if not data.get("data") or not data["data"].get("klines"):
        raise DataFetchError(f"No data for global index {index_code}")

    rows = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        rows.append({"date": parts[0], "value": float(parts[2])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    return df[["date", "value"]].reset_index(drop=True)


# ======================================================================
# Dispatch helpers
# ======================================================================

def fetch_fund(fund_type: str, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if fund_type == "a_share_mf":
        return fetch_a_share_mf(code, start_date, end_date)
    elif fund_type == "a_share_etf":
        return fetch_a_share_etf(code, start_date, end_date)
    elif fund_type == "overseas_etf":
        return fetch_overseas_etf(code, start_date, end_date)
    else:
        raise ValueError(f"Unknown fund type: {fund_type}")


