"""HTML report generation using Jinja2 templates."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - 基金组合分析报告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f6fa;
            color: #2d3436;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header {
            background: linear-gradient(135deg, #0984e3, #6c5ce7);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header .subtitle { opacity: 0.85; font-size: 14px; }
        .section {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .section h2 {
            font-size: 20px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #0984e3;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 14px;
        }
        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
            font-size: 12px;
            text-transform: uppercase;
        }
        tr:hover { background: #f8f9ff; }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .chart-container img { max-width: 100%; border-radius: 8px; }
        .warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 12px 16px;
            border-radius: 6px;
            margin: 8px 0;
            font-size: 13px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .metric-card .value {
            font-size: 24px;
            font-weight: 700;
            color: #0984e3;
        }
        .metric-card .label {
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }
        .footer {
            text-align: center;
            color: #aaa;
            font-size: 12px;
            margin-top: 40px;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }} · 组合分析报告</h1>
            <div class="subtitle">
                生成时间: {{ generated_at }} | {{ num_funds }} 只基金 |
                数据区间: {{ date_start }} ~ {{ date_end }}
            </div>
        </div>

        {% if warnings %}
        <div class="section">
            <div class="warning">
                ⚠️ 警告:<br>
                {% for w in warnings %}
                &nbsp;&nbsp;• {{ w }}<br>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- 组合概况 -->
        <div class="section">
            <h2>📊 组合概况</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="value">{{ summary.portfolio_return }}</div>
                    <div class="label">组合年化收益</div>
                </div>
                <div class="metric-card">
                    <div class="value">{{ summary.portfolio_volatility }}</div>
                    <div class="label">组合年化波动</div>
                </div>
                <div class="metric-card">
                    <div class="value">{{ summary.portfolio_sharpe }}</div>
                    <div class="label">夏普比率</div>
                </div>
                <div class="metric-card">
                    <div class="value">{{ summary.max_drawdown }}</div>
                    <div class="label">最大回撤</div>
                </div>
                <div class="metric-card">
                    <div class="value">{{ summary.effective_n }}</div>
                    <div class="label">有效持仓数</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>基金代码</th>
                        <th>名称</th>
                        <th>当前权重</th>
                        <th>建议权重(最大夏普)</th>
                        <th>建议权重(风险平价)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in weight_table %}
                    <tr>
                        <td>{{ row.code }}</td>
                        <td>{{ row.name }}</td>
                        <td>{{ row.current_weight }}</td>
                        <td>{{ row.ms_weight }}</td>
                        <td>{{ row.rp_weight }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- 风险收益指标 -->
        <div class="section">
            <h2>📈 风险收益指标</h2>
            {{ risk_table_html }}
        </div>

        {% if daily_advice_html %}
        <div class="section" style="border-left: 4px solid #e74c3c;">
            <h2>📋 今日操作建议 <span style="font-size:12px;color:#999;font-weight:normal;">（基于当日市场行情 + 组合数据）</span></h2>
            {{ daily_advice_html }}
        </div>
        {% endif %}

        {% if insights %}
        <div class="section">
            <h2>💡 分析结论与建议</h2>
            <div style="font-size:14px; line-height:2;">
            {{ insights }}
            </div>
        </div>
        {% endif %}

        <div class="footer">
            本报告由 Fund Portfolio Analyzer 自动生成 · 仅供参考，不构成投资建议
        </div>
    </div>
</body>
</html>
"""


def _build_insights(
    config, risk_metrics, corr_df, weight_table,
    ms_weights, rp_weights, neff, port_dd,
) -> str:
    """Generate plain-language analysis text from the numbers."""
    import numpy as np

    lines = []

    # 1. Diversification warning
    if neff < 3:
        lines.append(
            f"<b>⚠️ 分散度警报：</b>你持有 {len(config.funds)} 只基金，但有效持仓数仅 {neff}。"
            f"这意味着涨跌高度同步，分散效果很差。"
            f"主要原因是 A/C 份额重复持有，以及底层持仓大量重叠在科技/半导体。"
        )

    # 2. Best fund
    try:
        # Parse Sharpe from risk_metrics
        sharpes = {}
        for code, row in risk_metrics.iterrows():
            try:
                sharpes[code] = float(row["夏普比率"])
            except (ValueError, KeyError):
                pass
        if sharpes:
            best = max(sharpes, key=sharpes.get)
            worst = min(sharpes, key=sharpes.get)
            best_name = config.get_fund(best)
            best_name = best_name.name if best_name else best
            worst_name = config.get_fund(worst)
            worst_name = worst_name.name if worst_name else worst
            lines.append(
                f"<b>🏆 性价比最高：</b>{best_name}（夏普 {sharpes[best]:.2f}），"
                f"每承担 1% 风险能获得最高的收益。"
                f"当前仓位可能偏低，建议加仓。"
            )
            lines.append(
                f"<b>🔻 性价比最低：</b>{worst_name}（夏普 {sharpes[worst]:.2f}），"
                f"波动大但收益不成正比，建议减仓或替换。"
            )
    except Exception:
        pass

    # 3. Weight mismatch
    code_names = {f.code: f.name for f in config.funds}
    big_drift = []
    for wt in weight_table:
        cur = float(wt["current_weight"].rstrip("%")) / 100
        rp = float(wt["rp_weight"].rstrip("%")) / 100
        drift = abs(cur - rp)
        if drift > 0.05:
            direction = "太重" if cur > rp else "太轻"
            big_drift.append((wt["name"], cur, rp, direction))

    if big_drift:
        lines.append("<b>📐 仓位失衡：</b>以下基金当前仓位与风险平价建议差距超过 5%：")
        for name, cur, rp, direction in big_drift[:5]:
            lines.append(
                f"&nbsp;&nbsp;• {name}：当前 {cur:.0%}，建议 {rp:.0%}（{direction}）"
            )

    # 4. A/C merge reminder
    a_c_pairs = []
    seen = set()
    for f in config.funds:
        base = f.name.replace("A", "").replace("C", "")
        if base in seen:
            a_c_pairs.append(base)
        seen.add(base)
    if a_c_pairs:
        lines.append(
            "<b>🔄 A/C 份额重复：</b>你的组合中有同一基金同时持有 A 和 C 份额。"
            "两者的净值走势几乎一样，只是收费方式不同。建议每对只保留一个（长期持有选 A，短期选 C），"
            "可以节省管理费而不影响收益。"
        )

    # 5. Max drawdown warning
    dd_val = port_dd.get("max_drawdown", 0)
    if not isinstance(dd_val, (int, float)) or np.isnan(dd_val):
        dd_val = 0
    if abs(dd_val) > 0.15:
        tv = getattr(config, 'total_value', 100000)
        lines.append(
            f"<b>📉 最大回撤提醒：</b>组合整体最大回撤为 {dd_val:.1%}，"
            f"即 ¥{tv:,.0f} 最多跌到 ¥{tv*(1+dd_val):,.0f}。"
            f"如果你承受不了这么大的波动，需要加入债券或现金来缓冲。"
        )

    # 6. Cash suggestion
    lines.append(
        f"<b>💵 现金仓位建议：</b>当前市场半导体估值处于高位，"
        "机构普遍建议保留部分现金等待回调。建议留 10-20% 现金，等科技板块调整后再分批入场。"
    )

    return "<br>".join(lines)


def generate_html_report(
    config,
    nav_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    corr_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    max_sharpe_weights: "np.ndarray",
    risk_parity_weights: "np.ndarray",
    fund_names: dict[str, str],
    warnings: list[str],
    output_path: str | Path,
    daily_advice_html: str = "",
    chart_paths: list[str] | None = None,
) -> Path:
    """Generate a self-contained HTML report.

    Args:
        config: PortfolioConfig object.
        nav_df: Wide-format NAV DataFrame.
        daily_returns: Daily returns DataFrame.
        risk_metrics: Risk metrics table DataFrame.
        corr_df: Correlation matrix.
        portfolio_returns: Portfolio daily returns Series.
        max_sharpe_weights: Max Sharpe weight array.
        risk_parity_weights: Risk parity weight array.
        benchmark_results: Dict of benchmark analysis results.
        fund_names: Dict mapping code -> display name.
        warnings: List of warning strings.
        output_path: Path to save the HTML file.
        chart_paths: List of chart file paths (for logging).

    Returns:
        Path to the generated report.
    """
    import numpy as np
    from src.analysis.returns import annualized_return
    from src.analysis.risk import annualized_volatility, sharpe_ratio, max_drawdown
    from src.analysis.correlation import effective_n

    from jinja2 import Template

    # Portfolio summary metrics
    port_ret = annualized_return(portfolio_returns)
    port_vol = annualized_volatility(portfolio_returns)
    port_sharpe = sharpe_ratio(portfolio_returns, config.analysis.risk_free_rate)
    port_dd = max_drawdown(portfolio_returns)
    neff = effective_n(corr_df)

    summary = {
        "portfolio_return": f"{port_ret:.2%}",
        "portfolio_volatility": f"{port_vol:.2%}",
        "portfolio_sharpe": f"{port_sharpe:.2f}",
        "max_drawdown": f"{port_dd['max_drawdown']:.2%}" if not np.isnan(port_dd.get('max_drawdown', np.nan)) else "N/A",
        "effective_n": neff,
    }

    # Weight comparison table
    weight_table = []
    codes = list(daily_returns.columns)
    for i, code in enumerate(codes):
        current = config.weights.get(code, 0)
        ms = max_sharpe_weights[i] if i < len(max_sharpe_weights) else 0
        rp = risk_parity_weights[i] if i < len(risk_parity_weights) else 0
        weight_table.append({
            "code": code,
            "name": fund_names.get(code, code),
            "current_weight": f"{current:.1%}",
            "ms_weight": f"{ms:.1%}",
            "rp_weight": f"{rp:.1%}",
        })

    # Convert DataFrames to HTML
    risk_table_html = risk_metrics.to_html(
        classes="",
        border=0,
        justify="left",
        escape=False,
    )

    # ── Build plain-language insights ──────────────────────────────
    insights_html = _build_insights(
        config, risk_metrics, corr_df, weight_table,
        max_sharpe_weights, risk_parity_weights, neff, port_dd,
    )

    # Render template
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        title=config.name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        num_funds=len(nav_df.columns),
        date_start=str(nav_df.index[0]),
        date_end=str(nav_df.index[-1]),
        warnings=warnings,
        insights=insights_html,
        daily_advice_html=daily_advice_html,
        summary=summary,
        weight_table=weight_table,
        risk_table_html=risk_table_html,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    logger.info("Report saved to %s", output_path)
    return output_path
