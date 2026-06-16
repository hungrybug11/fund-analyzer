"""Format fund analysis results into WeCom Markdown messages.

Produces Markdown content compatible with 企业微信群机器人 API.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def build_markdown_message(
    portfolio_name: str,
    total_value: float,
    summary: dict[str, str],
    fund_count: int,
    date_range: tuple[str, str],
    advice_list: list[Any] | None = None,
    insights: str = "",
    report_path: str = "",
) -> str:
    """Build a complete WeCom Markdown message for portfolio analysis.

    Args:
        portfolio_name: Name of the portfolio.
        total_value: Total portfolio value in CNY.
        summary: Dict with keys: portfolio_return, portfolio_volatility,
                 portfolio_sharpe, max_drawdown, effective_n.
        fund_count: Number of funds held.
        date_range: (start, end) date tuple.
        advice_list: Optional list of FundAdvice dataclass instances.
        insights: Optional plain-text insights text.
        report_path: Optional path to local HTML report.

    Returns:
        Markdown string ready to send via WecomBot.send_markdown().
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_start, date_end = date_range
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────
    lines.append(f"# 📊 {portfolio_name} · 组合分析报告")
    lines.append("")
    lines.append(f"> 生成时间：{now}　｜　数据区间：{date_start} ~ {date_end}　｜　{fund_count} 只基金")
    lines.append("")

    # ── Key Metrics ─────────────────────────────────────────────────
    lines.append("## 📈 组合概况")
    lines.append("")
    lines.append(
        f"**年化收益**：{summary.get('portfolio_return', 'N/A')}　"
        f"**年化波动**：{summary.get('portfolio_volatility', 'N/A')}　"
        f"**夏普比率**：{summary.get('portfolio_sharpe', 'N/A')}"
    )
    lines.append(
        f"**最大回撤**：{summary.get('max_drawdown', 'N/A')}　"
        f"**有效持仓数**：{summary.get('effective_n', 'N/A')}　"
        f"**总市值**：¥{total_value:,.0f}"
    )
    lines.append("")

    # ── Daily Advice ────────────────────────────────────────────────
    if advice_list:
        lines.append("## 📋 今日操作建议")
        lines.append("")

        action_emojis = {
            "加仓": "📈", "减仓": "📉", "持有": "✅", "观望": "👀", "合并": "🔄",
            "buy": "📈", "sell": "📉", "hold": "✅", "reduce": "📉", "add": "📈",
        }

        for a in advice_list[:8]:
            action = getattr(a, "action", "")
            name = getattr(a, "name", "")
            urgency = getattr(a, "urgency", "")
            reason = getattr(a, "reason", "")
            emoji = action_emojis.get(action, "")
            urgency_tag = f"[{urgency}] " if urgency else ""
            line = f"{emoji} **{urgency_tag}{action}**　{name}"
            lines.append(line)
            if reason:
                lines.append(f"> {reason[:80]}")
            lines.append("")

        if len(advice_list) > 8:
            lines.append(f"> … 还有 {len(advice_list) - 8} 条建议略过")
            lines.append("")

    # ── Insights ────────────────────────────────────────────────────
    if insights:
        lines.append("## 💡 分析洞察")
        lines.append("")
        # Split by lines and add as markdown
        for para in insights.split("\n"):
            para = para.strip()
            if not para:
                continue
            # bold prefix detection
            lines.append(para)
            lines.append("")

    # ── Footer ──────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("_由 Fund Portfolio Analyzer 自动生成 · 仅供参考，不构成投资建议_")
    if report_path:
        lines.append(f"_完整报告：{report_path}_")

    return "\n".join(lines)
