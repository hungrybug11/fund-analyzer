"""Format fund analysis results into Feishu interactive card messages.

Creates a rich Feishu card (v4 template) that displays:
- Portfolio summary metrics
- Risk/return breakdown
- Daily advice actions
- Links to the full report
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def format_summary_card(
    portfolio_name: str,
    total_value: float,
    summary: dict[str, str],
    fund_count: int,
    date_range: tuple[str, str],
    insights: str = "",
    report_path: str = "",
) -> dict[str, Any]:
    """Build a Feishu interactive card showing the portfolio summary.

    Args:
        portfolio_name: Name of the portfolio.
        total_value: Total portfolio value in CNY.
        summary: Dict with keys: portfolio_return, portfolio_volatility,
                 portfolio_sharpe, max_drawdown, effective_n.
        fund_count: Number of funds held.
        date_range: (start_date, end_date) tuple.
        insights: Optional plain-text insights summary.
        report_path: Optional local path to the HTML report.

    Returns:
        A Feishu card JSON dict (msg_type="interactive").
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_start, date_end = date_range

    # ── Header ──────────────────────────────────────────────────────
    card: dict[str, Any] = {
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"📊 {portfolio_name} · 组合分析报告",
            },
            "template": "blue",
        },
        "elements": [
            # Subtitle
            {
                "tag": "note",
                "elements": [
                    {"tag": "plain_text", "content": f"生成时间: {now}  |  数据区间: {date_start} ~ {date_end}  |  {fund_count} 只基金"},
                ],
            },
            {"tag": "hr"},
        ],
    }

    # ── Key Metrics (2×3 grid using column_set) ────────────────────
    metrics = [
        ("组合年化收益", summary.get("portfolio_return", "N/A"), "green" if _is_positive(summary.get("portfolio_return", "0%")) else "red"),
        ("组合年化波动", summary.get("portfolio_volatility", "N/A"), "blue"),
        ("夏普比率", summary.get("portfolio_sharpe", "N/A"), "blue"),
        ("最大回撤", summary.get("max_drawdown", "N/A"), "red"),
        ("有效持仓数", summary.get("effective_n", "N/A"), "blue"),
        ("组合总市值", f"¥{total_value:,.0f}", "blue"),
    ]

    card["elements"].append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": "**📈 组合概况**",
        },
    })

    # Create two rows of 3 columns each
    for row_idx in range(0, len(metrics), 3):
        row_metrics = metrics[row_idx:row_idx + 3]
        columns = []
        for label, value, color in row_metrics:
            columns.append({
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "vertical_align": "center",
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**{value}**\n{label}",
                        },
                    },
                ],
            })

        card["elements"].append({
            "tag": "column_set",
            "flex_mode": "bisect",
            "background_style": "default",
            "columns": columns,
        })

    card["elements"].append({"tag": "hr"})

    return card


def format_risk_section(risk_metrics_html: str) -> dict[str, Any]:
    """Build a section showing risk metrics.  Not used directly —
       risk data is folded into other sections."""


def add_fund_performance_section(card: dict[str, Any], risk_rows: list[dict]) -> dict[str, Any]:
    """Append a fund-by-fund performance table to the card.

    Args:
        card: The card dict being built.
        risk_rows: List of dicts with keys: code, name, annual_return,
                   sharpe, max_drawdown, action (from advice).

    Returns:
        The updated card dict.
    """
    if not risk_rows:
        return card

    # Sort by Sharpe descending, take top 3 and bottom 1
    sorted_rows = sorted(risk_rows, key=lambda r: r.get("sharpe", 0), reverse=True)
    top = sorted_rows[:3]
    bottom = sorted_rows[-1:] if len(sorted_rows) > 3 else []

    lines = []
    if top:
        lines.append("**🏆 性价比前 3**")
        for r in top:
            name = r.get("name", r.get("code", ""))
            sharpe = r.get("sharpe", "N/A")
            ret = r.get("annual_return", "N/A")
            lines.append(f"• {name}  —  夏普 {sharpe}  年化 {ret}")

    if bottom:
        lines.append("")
        lines.append("**🔻 性价比最低**")
        for r in bottom:
            name = r.get("name", r.get("code", ""))
            sharpe = r.get("sharpe", "N/A")
            ret = r.get("annual_return", "N/A")
            lines.append(f"• {name}  —  夏普 {sharpe}  年化 {ret}")

    card["elements"].append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": "\n".join(lines),
        },
    })
    card["elements"].append({"tag": "hr"})
    return card


def add_daily_advice_section(card: dict[str, Any], advice_list: list[Any]) -> dict[str, Any]:
    """Append the daily trade advice to the card.

    Args:
        card: The card dict being built.
        advice_list: List of FundAdvice dataclass instances from daily_advice module.

    Returns:
        The updated card dict.
    """
    if not advice_list:
        return card

    # Actions mapping
    action_emojis = {
        "加仓": "📈",
        "减仓": "📉",
        "持有": "✅",
        "观望": "👀",
        "合并": "🔄",
        "buy": "📈",
        "sell": "📉",
        "hold": "✅",
        "reduce": "📉",
        "add": "📈",
    }

    lines = ["**📋 今日操作建议**"]

    for a in advice_list[:6]:  # top 6 to keep card readable
        action = getattr(a, "action", "")
        name = getattr(a, "name", "")
        urgency = getattr(a, "urgency", "")
        reason = getattr(a, "reason", "")
        emoji = action_emojis.get(action, "")
        urgency_tag = f"[{urgency}] " if urgency else ""
        lines.append(f"{emoji} {urgency_tag}{action:<4} {name}")
        if reason:
            lines.append(f"　└ {reason[:50]}")

    if len(advice_list) > 6:
        lines.append(f"…… 还有 {len(advice_list) - 6} 条建议")

    card["elements"].append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": "\n".join(lines),
        },
    })
    card["elements"].append({"tag": "hr"})
    return card


def add_quick_note(card: dict[str, Any], note: str) -> dict[str, Any]:
    """Append a small note (e.g. insights summary) at the bottom."""
    if not note:
        return card

    # Truncate to keep card tidy
    if len(note) > 300:
        note = note[:300] + "…"

    card["elements"].append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"💡 {note}",
        },
    })
    return card


def add_report_link(card: dict[str, Any], report_path: str) -> dict[str, Any]:
    """Append a button to open the full HTML report.

    Feishu cards can't link to local files directly, so we show a note.
    """
    if not report_path:
        return card

    card["elements"].append({
        "tag": "note",
        "elements": [
            {"tag": "plain_text", "content": f"📄 完整报告已保存至: {report_path}"},
        ],
    })
    return card


def build_portfolio_card(
    portfolio_name: str,
    total_value: float,
    summary: dict[str, str],
    fund_count: int,
    date_range: tuple[str, str],
    risk_rows: list[dict] | None = None,
    advice_list: list[Any] | None = None,
    insights: str = "",
    report_path: str = "",
) -> dict[str, Any]:
    """Build a complete portfolio analysis card.

    This is the main entry point — call this from run.py.

    Args:
        portfolio_name: Portfolio name.
        total_value: Total portfolio value.
        summary: Dict with 5 metric keys.
        fund_count: Number of funds.
        date_range: (start, end) date strings.
        risk_rows: Optional list of per-fund risk data.
        advice_list: Optional list of FundAdvice objects.
        insights: Optional insights text.
        report_path: Optional report file path.

    Returns:
        Complete Feishu card JSON dict.
    """
    card = format_summary_card(
        portfolio_name=portfolio_name,
        total_value=total_value,
        summary=summary,
        fund_count=fund_count,
        date_range=date_range,
        insights=insights,
        report_path=report_path,
    )

    if risk_rows:
        card = add_fund_performance_section(card, risk_rows)

    if advice_list:
        card = add_daily_advice_section(card, advice_list)

    if insights:
        card = add_quick_note(card, insights)

    if report_path:
        card = add_report_link(card, report_path)

    return card


def _is_positive(value_str: str) -> bool:
    """Check if a formatted percentage value is positive."""
    try:
        v = value_str.strip().rstrip("%")
        return float(v) > 0
    except (ValueError, AttributeError):
        return True
