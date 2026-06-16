"""Daily market-driven recommendations for each fund.

Combines:
  1. Current market sector trends (manually updated or fetched)
  2. Portfolio-level risk/return math (from our analysis)
  3. Fund-specific attributes (type, concentration, recent performance)

Returns per-fund action: 加仓 / 减仓 / 持有 / 观望, with reason and suggested amount.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# ======================================================================
# Market snapshot — update this section daily from news / jin10 / etc.
# ======================================================================

@dataclass
class SectorSignal:
    """One sector's current market signal."""
    name: str                    # e.g. "半导体", "QDII全球", "科技"
    trend: str                   # "强势" | "震荡" | "走弱"
    note: str                    # one-line reason
    sentiment: float = 0.0       # -1.0 (bearish) to +1.0 (bullish)


TODAY = date.today().strftime("%Y-%m-%d")

MARKET_SNAPSHOT: dict[str, SectorSignal] = {
    "半导体": SectorSignal(
        name="半导体",
        trend="强势",
        note="半导体板块资金净流入+163亿，全行业第1。永赢半导体今日+5.68%，近6月+93.76%。主力大幅加仓，板块热度极高。",
        sentiment=0.8,
    ),
    "QDII全球": SectorSignal(
        name="QDII全球",
        trend="震荡偏强",
        note="美股标普+0.50%、纳指+0.31%。易方达全球优质近1年+86%，同类排名21。QDII限购普遍，额度稀缺推升溢价。",
        sentiment=0.3,
    ),
    "科技": SectorSignal(
        name="科技",
        trend="强势",
        note="永赢科技今日+3.62%，近1年+307%，同类第21。消费电子资金流入+47亿。科技板块内部轮动健康，AI+硬件双主线支撑。",
        sentiment=0.6,
    ),
    "通信": SectorSignal(
        name="通信",
        trend="强势",
        note="通信设备板块资金净流入+147亿，全行业第2。东财通信今日+7.36%。5G/光通信/卫星互联网多重催化。",
        sentiment=0.7,
    ),
    "混合": SectorSignal(
        name="混合",
        trend="强势",
        note="信澳业绩驱动今日+8.59%，近1年+357%同类第7。大盘普涨（上证+1.61%、深证+3.79%），主力净流入+685亿，北向+840亿。",
        sentiment=0.6,
    ),
}

# ======================================================================
# Fund → sector mapping
# ======================================================================

FUND_SECTOR_MAP: dict[str, str] = {
    "025208": "半导体",       # 永赢半导体A
    "018229": "QDII全球",     # 易方达全球优质A
    "022364": "科技",         # 永赢科技A
    "012920": "QDII全球",     # 易方达全球成长精选A
    "017731": "QDII全球",     # 嘉实产业升级C
    "016665": "QDII全球",     # 天弘全球高端制造C
    "017730": "QDII全球",     # 嘉实产业升级A
    "012922": "QDII全球",     # 易方达全球成长精选C
    "016664": "QDII全球",     # 天弘全球高端制造A
    "008326": "通信",         # 东财通信A
    "006502": "半导体",       # 财通集成电路产业A
}


# ======================================================================
# Recommendation engine
# ======================================================================

@dataclass
class FundAdvice:
    code: str
    name: str
    action: str            # "加仓" | "减仓" | "持有" | "观望" | "清仓"
    urgency: str           # "🔴 立即" | "🟡 本周" | "🟢 等待"
    reason: str
    suggested_amount: str  # e.g. "¥3,000-5,000" or "不操作"
    risk_note: str = ""


def _find_c_partner(a_code: str, a_name: str, fund_configs, weights) -> str | None:
    """Find C-share partner for an A-share fund by name pattern."""
    if not a_name.endswith("A"):
        return None
    c_name = a_name[:-1] + "C"
    for f in fund_configs:
        if f.name == c_name and f.code in weights:
            return f.code
    return None


def generate_daily_advice(
    fund_configs: list,
    daily_returns: pd.DataFrame,
    weights: dict[str, float],
    total_value: float = 100000,
) -> list[FundAdvice]:
    """Generate per-fund advice for today.

    Decision logic (in priority order):
      1. A/C duplicate → suggest merge
      2. Sector bearish + fund overweight → reduce
      3. Sector bullish + fund underweight → add
      4. High Sharpe + not at max → gradually add
      5. Low Sharpe + high weight → gradually reduce
      6. Everything else → hold
    """
    from src.analysis.risk import sharpe_ratio, max_drawdown

    advice_list = []
    today = TODAY

    # Pre-compute per-fund stats
    fund_stats = {}
    for f in fund_configs:
        if f.code in daily_returns.columns:
            series = daily_returns[f.code].dropna()
            if len(series) > 20:
                fund_stats[f.code] = {
                    "sharpe": sharpe_ratio(series),
                    "max_dd": max_drawdown(series).get("max_drawdown", 0),
                    "vol": series.std() * np.sqrt(252),
                    "ret": series.mean() * 252,
                }

    for f in fund_configs:
        code = f.code
        name = f.name
        weight = weights.get(code, 0)
        sector_key = FUND_SECTOR_MAP.get(code, "混合")
        sector = MARKET_SNAPSHOT.get(sector_key)
        stats = fund_stats.get(code, {})

        sharpe_val = stats.get("sharpe", np.nan)
        max_dd_val = stats.get("max_dd", 0)

        if sector is None:
            sector = MARKET_SNAPSHOT.get("混合")

        sector_sentiment = sector.sentiment if sector else 0.0

        # ---- A/C duplicate detection ----
        is_a = name.endswith("A")
        is_c = name.endswith("C")

        if is_c:
            # C shares: suggest merge, OR reduce+merge if sector bad
            if sector_sentiment < -0.3 and weight > 0.12:
                advice_list.append(FundAdvice(
                    code=code, name=name,
                    action="减仓+合并",
                    urgency="🔴 立即",
                    reason=f"{sector_key}板块{sector.trend}，当前仓位 {weight:.0%} 过重。"
                           f"建议先减仓一半再转到A份额。{sector.note[:50]}…",
                    suggested_amount="先赎回一半，剩余转A份额",
                    risk_note="风格切换风险",
                ))
            else:
                advice_list.append(FundAdvice(
                    code=code, name=name,
                    action="合并",
                    urgency="🟡 本周",
                    reason="C份额持有成本高于A份额。长期持有应转到A，省0.4%/年的销售服务费。走势一样，不影响收益。",
                    suggested_amount="全部转到A份额",
                    risk_note="",
                ))
            continue

        if is_a:
            c_code = _find_c_partner(code, name, fund_configs, weights)
            if c_code is not None:
                merged_weight = weight + weights.get(c_code, 0)
                if sector_sentiment < -0.3 and merged_weight > 0.20:
                    advice_list.append(FundAdvice(
                        code=code, name=name,
                        action="合并后减仓",
                        urgency="🔴 立即",
                        reason=f"C份额合并后总仓位将达 {merged_weight:.0%}，"
                               f"而{sector_key}板块{sector.trend}。建议合并完成后减仓到 10-12%。",
                        suggested_amount="合并后赎回一半",
                        risk_note="仓位过重+板块走弱",
                    ))
                else:
                    advice_list.append(FundAdvice(
                        code=code, name=name,
                        action="接收C合并",
                        urgency="🟡 本周",
                        reason="同基金的C份额应合并到这里。合并后仓位会增加，请重新评估比例。",
                        suggested_amount="接收C份额转入",
                        risk_note="",
                    ))
                continue
            # No C partner → fall through to sector logic

        # ---- Sector-driven decision ----

        # High Sharpe + underweight in neutral-or-better sector → add
        if (not np.isnan(sharpe_val) and sharpe_val > 4.0
                and weight < 0.12 and sector_sentiment >= -0.3):
            target_w = min(0.15, weight + 0.05)
            add_pct = target_w - weight
            add_amt = int(total_value * add_pct / 1000) * 1000
            advice_list.append(FundAdvice(
                code=code, name=name,
                action="加仓",
                urgency="🟡 本周" if sector_sentiment < 0 else "🟢 等待回调",
                reason=f"夏普比率 {sharpe_val:.1f}（组合最高档），当前仅占 {weight:.0%}。"
                       f"建议提到 {target_w:.0%} 左右。{sector.note[:50]}…",
                suggested_amount=f"¥{add_amt:,}（分2-3次加）",
                risk_note=f"当前最大回撤 {max_dd_val:.1%}" if not np.isnan(max_dd_val) else "",
            ))
            continue

        # Bearish sector + overweight → reduce
        if sector_sentiment < -0.3 and weight > 0.12:
            target_w = max(0.05, weight - 0.08)
            reduce_pct = weight - target_w
            reduce_amt = int(total_value * reduce_pct / 1000) * 1000
            advice_list.append(FundAdvice(
                code=code, name=name,
                action="减仓",
                urgency="🔴 立即" if sector_sentiment < -0.5 else "🟡 本周",
                reason=f"{sector_key}板块{sector.trend}。{sector.note[:60]}…"
                       f"当前仓位 {weight:.0%}，建议降到 {target_w:.0%}。"
                       f"先锁利，等企稳再接回。",
                suggested_amount=f"赎回 ¥{reduce_amt:,}",
                risk_note="风格切换中，不止损可能扩大回撤",
            ))
            continue

        # Bearish sector + small weight → hold/wait
        if sector_sentiment < -0.3 and weight <= 0.12:
            advice_list.append(FundAdvice(
                code=code, name=name,
                action="观望",
                urgency="🟢 等待",
                reason=f"{sector_key}板块走弱，但你的仓位不大（{weight:.0%}），不用急着卖。"
                       f"等板块企稳后再判断。",
                suggested_amount="不操作",
                risk_note=f"止损线建议设在回撤-15%",
            ))
            continue

        # Bullish sector + adequate weight → hold
        if sector_sentiment > 0 and 0.05 <= weight <= 0.20:
            advice_list.append(FundAdvice(
                code=code, name=name,
                action="持有",
                urgency="🟢 持有",
                reason=f"{sector_key}板块趋势向好。仓位 {weight:.0%} 合理，继续拿着。{sector.note[:40]}",
                suggested_amount="不操作",
                risk_note="",
            ))
            continue

        # Neutral / default → hold
        advice_list.append(FundAdvice(
            code=code, name=name,
            action="持有",
            urgency="🟢 持有",
            reason=f"当前仓位 {weight:.0%}，板块信号中性。跟踪观察，不急于操作。",
            suggested_amount="不操作",
            risk_note="",
        ))

    return advice_list


def advice_to_dataframe(advice_list: list[FundAdvice]) -> pd.DataFrame:
    """Convert advice list to DataFrame for display."""
    rows = []
    for a in advice_list:
        rows.append({
            "基金": a.name,
            "代码": a.code,
            "操作": a.action,
            "优先级": a.urgency,
            "理由": a.reason,
            "建议金额": a.suggested_amount,
        })
    return pd.DataFrame(rows)


def advice_to_html(advice_list: list[FundAdvice]) -> str:
    """Render advice as HTML for the report."""
    import datetime
    today_str = TODAY

    html = f'<p style="color:#888;font-size:13px;">📅 建议日期：{today_str}（基于当日市场行情）</p>'
    html += '<table style="font-size:13px; width:100%; border-collapse:collapse;">'
    html += '<tr style="background:#f0f0f0;"><th>基金</th><th>操作</th><th>理由</th><th>金额</th></tr>'

    for a in advice_list:
        action_color = {
            "加仓": "#27ae60", "减仓": "#e74c3c", "持有": "#7f8c8d",
            "观望": "#f39c12", "合并": "#3498db", "接收C合并": "#3498db",
            "清仓": "#c0392b",
        }.get(a.action, "#333")

        html += f'<tr>'
        html += f'<td style="max-width:180px;">{a.name}</td>'
        html += f'<td style="color:{action_color};font-weight:bold;">{a.urgency} {a.action}</td>'
        html += f'<td style="max-width:350px;font-size:12px;">{a.reason}</td>'
        html += f'<td style="white-space:nowrap;">{a.suggested_amount}</td>'
        html += f'</tr>'

    html += '</table>'

    # Market summary
    html += '<br><p style="font-size:12px;color:#888;"><b>📊 今日市场速览：</b><br>'
    for key, sig in MARKET_SNAPSHOT.items():
        emoji = {"强势": "🟢", "震荡": "🟡", "震荡偏弱": "🟠", "走弱": "🔴"}.get(sig.trend, "⚪")
        html += f'&nbsp;&nbsp;{emoji} <b>{key}</b>：{sig.trend} — {sig.note[:80]}…<br>'
    html += '</p>'

    return html
