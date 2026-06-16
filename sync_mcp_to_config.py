#!/usr/bin/env python
"""Sync MCP portfolio.json to config.yaml.

Reads the MCP portfolio (shares + costPrice), fetches current NAV,
computes amounts and weights, and writes config.yaml.

Usage: python sync_mcp_to_config.py
"""
import json
import urllib.request
from pathlib import Path

MCP_PORTFOLIO = Path.home() / "cn-funds-mcp" / "data" / "portfolio.json"
CONFIG_PATH = Path(__file__).parent / "config.yaml"

FUND_META = {
    "018229": ("易方达全球优质企业混合(QDII)A", "a_share_mf", "^GSPC"),
    "025208": ("永赢先锋半导体智选混合A", "a_share_mf", "000300"),
    "022364": ("永赢科技智选混合A", "a_share_mf", "000300"),
    "012920": ("易方达全球成长精选混合(QDII)A", "a_share_mf", "^GSPC"),
    "017731": ("嘉实全球产业升级股票(QDII)C", "a_share_mf", "^GSPC"),
    "016665": ("天弘全球高端制造混合(QDII)C", "a_share_mf", "^GSPC"),
    "017730": ("嘉实全球产业升级股票(QDII)A", "a_share_mf", "^GSPC"),
    "012922": ("易方达全球成长精选混合(QDII)C", "a_share_mf", "^GSPC"),
    "016664": ("天弘全球高端制造混合(QDII)A", "a_share_mf", "^GSPC"),
    "008326": ("东财通信A", "a_share_mf", "000300"),
    "006502": ("财通集成电路产业股票A", "a_share_mf", "000300"),
}


def get_current_nav(fund_codes):
    """Fetch current NAV for a list of fund codes."""
    codes_str = ",".join(fund_codes)
    url = (
        f"https://fundmobapi.eastmoney.com/FundMNewApi/FundMNFInfo"
        f"?pageIndex=1&pageSize=200&plat=Android&appType=ttjj"
        f"&product=EFund&Version=1&deviceid=Wap&Fcodes={codes_str}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://fund.eastmoney.com/",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return {d["FCODE"]: float(d["NAV"]) for d in data.get("Datas", [])}


def sync():
    if not MCP_PORTFOLIO.exists():
        print(f"MCP portfolio not found at {MCP_PORTFOLIO}")
        return

    with open(MCP_PORTFOLIO, "r", encoding="utf-8") as f:
        portfolio = json.load(f)

    funds = portfolio.get("funds", [])
    if not funds:
        print("MCP portfolio is empty")
        return

    codes = [f["fundCode"] for f in funds]
    print(f"Fetching NAV for {len(codes)} funds...")
    navs = get_current_nav(codes)

    amounts = {}
    for f in funds:
        code = f["fundCode"]
        nav = navs.get(code)
        if nav:
            amounts[code] = round(f["shares"] * nav, 2)
        else:
            print(f"  WARNING: no NAV for {code}, using cost")
            amounts[code] = round(f["shares"] * f["costPrice"], 2)

    total = sum(amounts.values())

    lines = []
    lines.append("# ============================================================")
    lines.append("#  Fund Portfolio Analysis Configuration")
    lines.append(f"#  {len(funds)} funds, total value {total:,.0f}")
    lines.append("#  Auto-generated from MCP portfolio.json")
    lines.append("# ============================================================")
    lines.append("")
    lines.append("portfolio:")
    lines.append("  name: 我的基金组合")
    lines.append("  base_currency: CNY")
    lines.append(f"  total_value: {total:.0f}")
    lines.append("")
    lines.append("funds:")

    for f in funds:
        code = f["fundCode"]
        amt = amounts.get(code, 0)
        w = amt / total if total > 0 else 0
        name, ftype, bm = FUND_META.get(
            code, (f.get("name", code), "a_share_mf", "000300")
        )
        lines.append(f'  - code: "{code}"')
        lines.append(f'    name: "{name}"')
        lines.append(f'    type: "{ftype}"')
        lines.append(f"    weight: {w:.4f}")
        lines.append(f'    benchmark: "{bm}"')
        lines.append(f'    currency: "CNY"')
        lines.append("")

    lines.append("analysis:")
    lines.append("  risk_free_rate: 0.025")
    lines.append("  lookback_years: 5")
    lines.append("  rebalance_frequency: quarterly")
    lines.append("  rebalance_band: 0.05")
    lines.append("")
    lines.append("output:")
    lines.append("  data_dir: data")
    lines.append("  output_dir: output")
    lines.append("  chart_format: html")
    lines.append("  generate_html_report: true")
    lines.append("  cache_expiry_days: 1")
    lines.append("")
    lines.append("notify:")
    lines.append("  feishu_enabled: false")
    lines.append("  feishu_webhook_url: ''")
    lines.append("  wecom_enabled: false")
    lines.append("  wecom_webhook_url: ''")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Synced: {len(funds)} funds, total = {total:,.0f}")
    for f in funds:
        code = f["fundCode"]
        amt = amounts.get(code, 0)
        name = FUND_META.get(code, (f.get("name", ""),))[0]
        print(f"  {code}  {name}:  {amt:,.0f}  ({amt/total*100:.1f}%)")


if __name__ == "__main__":
    sync()
