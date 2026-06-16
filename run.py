#!/usr/bin/env python
"""Fund Portfolio Analyzer — Single entry point.

Usage:
    python run.py              # Full pipeline: fetch → analyze → report
    python run.py --fetch-only # Only fetch/update data cache
    python run.py --config my_config.yaml  # Use a custom config file

Edit config.yaml to add your fund holdings before running.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Fix Unicode output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, PortfolioConfig
from src.data.manager import get_portfolio_data
from src.analysis.returns import (
    compute_returns,
    compute_portfolio_returns,
    annualized_return,
)
from src.analysis.risk import (
    risk_metrics_table,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
)
from src.analysis.correlation import correlation_matrix, hierarchical_clustering, effective_n
from src.analysis.optimization import (
    expected_returns,
    covariance_matrix,
    efficient_frontier,
    max_sharpe_portfolio,
    risk_parity_portfolio,
    min_variance_portfolio,
)
from src.reporting.charts import save_all_charts
from src.reporting.report import generate_html_report
from src.analysis.daily_advice import generate_daily_advice, advice_to_dataframe, advice_to_html
from src.notify.feishu_bot import FeishuBot, FeishuError
from src.notify.wechat_bot import WecomBot, WecomError
from src.notify import feishu_formatter, wechat_formatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fund_analyzer")


def run_pipeline(config_path: str = "config.yaml", fetch_only: bool = False):
    """Execute the full analysis pipeline.

    Args:
        config_path: Path to config.yaml.
        fetch_only: If True, only fetch/cache data without analysis.
    """
    # ── 1. Load config ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Fund Portfolio Analyzer")
    logger.info("=" * 60)

    try:
        config = load_config(config_path)
        logger.info("Loaded config: %s (%d funds)", config.name, len(config.funds))
    except (FileNotFoundError, ValueError) as e:
        logger.error("Config error: %s", e)
        sys.exit(1)

    # ── 2. Fetch data ───────────────────────────────────────────────
    logger.info("Fetching portfolio data...")
    try:
        nav_df, nav_display, warnings = get_portfolio_data(config)
        logger.info("Portfolio data: %d funds x %d dates", len(nav_df.columns), len(nav_df))
    except RuntimeError as e:
        logger.error("Data fetch failed: %s", e)
        sys.exit(1)

    if fetch_only:
        logger.info("Data fetch complete. Cache updated. Exiting.")
        return

    # ── 3. Compute returns ───────────────────────────────────────────
    daily_returns = compute_returns(nav_df)
    port_returns = compute_portfolio_returns(nav_df, config.weights)

    # Build fund name lookup
    fund_names = {f.code: f.name for f in config.funds}

    # ── 4. Risk analysis ─────────────────────────────────────────────
    logger.info("Computing risk metrics...")
    risk_table = risk_metrics_table(daily_returns, config.analysis.risk_free_rate)
    # Format for display
    pct_cols = ["年化收益率", "年化波动率", "最大回撤", "VaR 95%", "CVaR 95%"]
    ratio_cols = ["夏普比率", "索提诺比率", "卡玛比率"]
    risk_display = risk_table.copy()
    for c in pct_cols:
        if c in risk_display.columns:
            risk_display[c] = risk_display[c].apply(lambda x: f"{x:.2%}" if not np.isnan(x) else "N/A")
    for c in ratio_cols:
        if c in risk_display.columns:
            risk_display[c] = risk_display[c].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
    print("\n" + "=" * 60)
    print("  Risk & Return Metrics")
    print("=" * 60)
    print(risk_display.to_string())

    # ── 5. Correlation analysis ──────────────────────────────────────
    logger.info("Computing correlations...")
    corr_df = correlation_matrix(daily_returns)
    linkage, labels = hierarchical_clustering(daily_returns)
    neff = effective_n(corr_df)
    print(f"\nEffective N (diversification): {neff} of {len(corr_df)} funds")

    # ── 6. Portfolio optimization ────────────────────────────────────
    logger.info("Running portfolio optimization...")
    exp_ret = expected_returns(daily_returns)
    cov = covariance_matrix(daily_returns)
    rf = config.analysis.risk_free_rate

    frontier = efficient_frontier(exp_ret, cov, rf)
    ms_weights = max_sharpe_portfolio(exp_ret, cov, rf)
    rp_weights = risk_parity_portfolio(cov)
    mv_weights = min_variance_portfolio(cov)

    print("\n--- Optimal Weights ---")
    print(f"{'Fund':<12} {'Current':>8} {'Max Sharpe':>12} {'Risk Parity':>12} {'Min Var':>12}")
    print("-" * 56)
    for i, code in enumerate(daily_returns.columns):
        name = fund_names.get(code, code)
        cur = config.weights.get(code, 0)
        print(f"{name:<12} {cur:>8.1%} {ms_weights[i]:>12.1%} {rp_weights[i]:>12.1%} {mv_weights[i]:>12.1%}")

    # -- 7. Daily advice --
    logger.info("Generating daily advice...")
    advice_list = generate_daily_advice(config.funds, daily_returns, config.weights, config.total_value)
    advice_html = advice_to_html(advice_list)
    print("\n--- 今日操作建议 ---")
    for a in advice_list:
        print(f"{a.urgency} {a.action:<4} | {a.name:<20} | {a.reason[:60]}")

    # -- 8. Generate charts --
    logger.info("Generating charts...")
    output_dir = Path(config.output.output_dir) / "charts"
    chart_paths = save_all_charts(
        nav_df=nav_df,
        daily_returns=daily_returns,
        corr_df=corr_df,
        linkage_result=linkage,
        labels=labels,
        frontier=frontier,
        max_sharpe_weights=ms_weights,
        risk_parity_weights=rp_weights,
        min_var_weights=mv_weights,
        exp_ret=exp_ret,
        cov=cov,
        portfolio_returns=port_returns,
        fund_names=fund_names,
        output_dir=output_dir,
        chart_format=config.output.chart_format,
        nav_display=nav_display,
    )

    # ── 9. Generate HTML report ─────────────────────────────────────
    if config.output.generate_html_report:
        logger.info("Generating HTML report...")
        report_path = generate_html_report(
            config=config,
            nav_df=nav_df,
            daily_returns=daily_returns,
            risk_metrics=risk_table,
            corr_df=corr_df,
            portfolio_returns=port_returns,
            max_sharpe_weights=ms_weights,
            risk_parity_weights=rp_weights,
            fund_names=fund_names,
            warnings=warnings,
            output_path=Path(config.output.output_dir) / "report.html",
            daily_advice_html=advice_html,
            chart_paths=[str(p) for p in chart_paths],
        )
        print(f"\n Report saved to: {report_path}")

    # -- 10. Send notifications (if enabled) --
    if config.notify.feishu_enabled or config.notify.wecom_enabled:
        # Compute shared summary metrics once
        port_ret = annualized_return(port_returns)
        port_vol = annualized_volatility(port_returns)
        port_sharpe = sharpe_ratio(port_returns, config.analysis.risk_free_rate)
        port_dd_info = max_drawdown(port_returns)
        neff = effective_n(corr_df)

        summary = {
            "portfolio_return": f"{port_ret:.2%}",
            "portfolio_volatility": f"{port_vol:.2%}",
            "portfolio_sharpe": f"{port_sharpe:.2f}",
            "max_drawdown": (
                f"{port_dd_info['max_drawdown']:.2%}"
                if isinstance(port_dd_info.get("max_drawdown"), (int, float))
                and not __import__("math").isnan(port_dd_info["max_drawdown"])
                else "N/A"
            ),
            "effective_n": f"{neff:.1f}",
        }

        report_path_str = str(report_path) if config.output.generate_html_report else ""

        # -- 10a. Feishu --
        if config.notify.feishu_enabled and config.notify.feishu_webhook_url:
            logger.info("Sending Feishu notification...")
            try:
                risk_rows = []
                for code, row in risk_table.iterrows():
                    fund = config.get_fund(code)
                    risk_rows.append({
                        "code": code,
                        "name": fund.name if fund else code,
                        "annual_return": row.get("年化收益率", "N/A"),
                        "sharpe": row.get("夏普比率", "N/A"),
                        "max_drawdown": row.get("最大回撤", "N/A"),
                    })

                card = feishu_formatter.build_portfolio_card(
                    portfolio_name=config.name,
                    total_value=config.total_value,
                    summary=summary,
                    fund_count=len(config.funds),
                    date_range=(str(nav_df.index[0]), str(nav_df.index[-1])),
                    risk_rows=risk_rows,
                    advice_list=advice_list,
                    insights="",
                    report_path=report_path_str,
                )
                bot = FeishuBot(config.notify.feishu_webhook_url)
                bot.send_card(card)
                print(" Feishu notification sent!")
            except (FeishuError, Exception) as e:
                logger.warning("Feishu notification failed: %s", e)
                print(f" Feishu notification failed: {e}")

        # -- 10b. WeCom (企业微信) --
        if config.notify.wecom_enabled and config.notify.wecom_webhook_url:
            logger.info("Sending WeCom notification...")
            try:
                md = wechat_formatter.build_markdown_message(
                    portfolio_name=config.name,
                    total_value=config.total_value,
                    summary=summary,
                    fund_count=len(config.funds),
                    date_range=(str(nav_df.index[0]), str(nav_df.index[-1])),
                    advice_list=advice_list,
                    insights="",  # insights text can be generated if needed
                    report_path=report_path_str,
                )
                bot = WecomBot(config.notify.wecom_webhook_url)
                bot.send_markdown(md)
                print(" WeCom notification sent!")
            except (WecomError, Exception) as e:
                logger.warning("WeCom notification failed: %s", e)
                print(f" WeCom notification failed: {e}")

    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print("=" * 60)

    if warnings:
        print("\n Warnings:")
        for w in warnings:
            print(f"  - {w}")


def main():
    parser = argparse.ArgumentParser(
        description="Fund Portfolio Analyzer — quantitative analysis for fund holdings."
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch/cache data, skip analysis",
    )
    args = parser.parse_args()
    run_pipeline(config_path=args.config, fetch_only=args.fetch_only)


if __name__ == "__main__":
    main()
