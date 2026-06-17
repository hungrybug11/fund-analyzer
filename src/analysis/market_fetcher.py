# Add to daily_advice.py - auto-fetch market snapshot via EastMoney API

import json
import logging
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SectorSignal:
    name: str
    trend: str
    note: str
    sentiment: float = 0.0


def _fetch_json(url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.eastmoney.com/",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def fetch_market_snapshot() -> dict[str, SectorSignal]:
    """Fetch real-time market data from EastMoney and build sector signals."""
    
    # 1. Get market overview
    try:
        url_idx = "https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&secids=1.000001,0.399001&fields=f2,f3,f4,f104,f105,f106"
        idx_data = _fetch_json(url_idx)
        diffs = idx_data.get("data", {}).get("diff", [])
        sh = diffs[0] if len(diffs) > 0 else {}
        sz = diffs[1] if len(diffs) > 1 else {}
        sh_pct = sh.get("f3", 0)
        sz_pct = sz.get("f3", 0)
        up_total = sh.get("f104", 0) + sz.get("f104", 0)
        down_total = sh.get("f105", 0) + sz.get("f105", 0)
        
        if sh_pct > 0.5 and sz_pct > 0.5 and up_total > down_total:
            market_mood = "普涨"
            market_sentiment = 0.5
        elif sh_pct < -0.5 and sz_pct < -0.5:
            market_mood = "普跌"
            market_sentiment = -0.5
        else:
            market_mood = "分化"
            market_sentiment = 0.0
    except Exception as e:
        logger.warning("Failed to fetch market overview: %s", e)
        sh_pct = sz_pct = 0
        up_total = down_total = 0
        market_mood = "未知"
        market_sentiment = 0.0

    # 2. Get sector capital flow
    try:
        url_sector = "https://data.eastmoney.com/dataapi/bkzj/getbkzj?key=f62&code=m%3A90%2Bs%3A4"
        sector_data = _fetch_json(url_sector)
        sector_flows = {}
        for d in sector_data.get("data", {}).get("diff", []):
            sector_flows[d["f14"]] = d["f62"] / 1e8  # convert to billions
    except Exception as e:
        logger.warning("Failed to fetch sector flows: %s", e)
        sector_flows = {}

    # 3. Build sector signals
    def _flow_signal(name: str, flow: float) -> tuple[str, float]:
        if flow > 30:
            return "强势 🔥", 0.7
        elif flow > 10:
            return "偏强", 0.4
        elif flow > 0:
            return "震荡偏强", 0.2
        elif flow > -10:
            return "震荡", 0.0
        elif flow > -20:
            return "偏弱", -0.3
        else:
            return "走弱", -0.6

    # Map sector names from EastMoney to our sector keys
    sector_map = {
        "半导体": "半导体",
        "通信设备": "通信",
        "消费电子": "科技",
        "元件": "科技",
    }

    signals = {}
    for em_name, our_key in sector_map.items():
        flow = sector_flows.get(em_name, 0)
        trend, sentiment = _flow_signal(em_name, flow)
        signals[our_key] = SectorSignal(
            name=our_key,
            trend=trend,
            note=f"{em_name}资金净流入{flow:+.1f}亿",
            sentiment=sentiment,
        )

    # QDII (no real-time sector data, use market mood proxy)
    qdii_sentiment = market_sentiment * 0.6
    qdii_trend = "偏强" if qdii_sentiment > 0.1 else ("偏弱" if qdii_sentiment < -0.1 else "震荡")
    signals["QDII全球"] = SectorSignal(
        name="QDII全球",
        trend=qdii_trend,
        note=f"A股{market_mood}（上证{sh_pct:+.2f}% 深证{sz_pct:+.2f}%，上涨{up_total}下跌{down_total}）",
        sentiment=qdii_sentiment,
    )

    # Mixed/other
    signals["混合"] = SectorSignal(
        name="混合",
        trend="震荡" if abs(market_sentiment) < 0.3 else ("偏强" if market_sentiment > 0 else "偏弱"),
        note=f"市场{market_mood}，涨跌比{up_total}:{down_total}",
        sentiment=market_sentiment * 0.5,
    )

    return signals
