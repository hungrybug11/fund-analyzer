"""Configuration loader and validator.

Reads config.yaml and returns a validated PortfolioConfig object.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class FundConfig:
    code: str
    name: str
    type: str          # "a_share_mf" | "a_share_etf" | "overseas_etf"
    weight: float
    currency: str


@dataclass
class AnalysisParams:
    risk_free_rate: float = 0.025
    lookback_years: int = 5
    rebalance_frequency: str = "quarterly"
    rebalance_band: float = 0.05


@dataclass
class OutputConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    chart_format: str = "png"
    generate_html_report: bool = True
    cache_expiry_days: int = 1


@dataclass
class NotifyConfig:
    """Configuration for sending notifications."""
    feishu_enabled: bool = False
    feishu_webhook_url: str = ""
    wecom_enabled: bool = False
    wecom_webhook_url: str = ""


@dataclass
class PortfolioConfig:
    name: str
    base_currency: str
    funds: list[FundConfig]
    analysis: AnalysisParams
    output: OutputConfig
    total_value: float = 100000
    notify: NotifyConfig = field(default_factory=NotifyConfig)

    @property
    def fund_codes(self) -> list[str]:
        return [f.code for f in self.funds]

    @property
    def weights(self) -> dict[str, float]:
        return {f.code: f.weight for f in self.funds}

    def get_fund(self, code: str) -> Optional[FundConfig]:
        for f in self.funds:
            if f.code == code:
                return f
        return None


def load_config(config_path: str | Path = "config.yaml") -> PortfolioConfig:
    """Load and validate the YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Validated PortfolioConfig object.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If configuration is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("Config file is empty.")

    # Parse portfolio section
    pf = raw.get("portfolio", {})
    name = pf.get("name", "My Portfolio")
    base_currency = pf.get("base_currency", "CNY")
    total_value = float(pf.get("total_value", 100000))

    # Parse funds
    funds = []
    for f in raw.get("funds", []):
        funds.append(FundConfig(
            code=str(f["code"]),
            name=f.get("name", f["code"]),
            type=f.get("type", "a_share_mf"),
            weight=float(f.get("weight", 0)),
            currency=f.get("currency", "CNY"),
        ))

    # Parse analysis params
    a = raw.get("analysis", {})
    analysis = AnalysisParams(
        risk_free_rate=float(a.get("risk_free_rate", 0.025)),
        lookback_years=int(a.get("lookback_years", 5)),
        rebalance_frequency=str(a.get("rebalance_frequency", "quarterly")),
        rebalance_band=float(a.get("rebalance_band", 0.05)),
    )

    # Parse output config
    o = raw.get("output", {})
    output = OutputConfig(
        data_dir=str(o.get("data_dir", "data")),
        output_dir=str(o.get("output_dir", "output")),
        chart_format=str(o.get("chart_format", "png")),
        generate_html_report=bool(o.get("generate_html_report", True)),
        cache_expiry_days=int(o.get("cache_expiry_days", 1)),
    )

    # Parse notify config
    n = raw.get("notify", {})
    notify = NotifyConfig(
        feishu_enabled=bool(n.get("feishu_enabled", False)),
        feishu_webhook_url=str(n.get("feishu_webhook_url", "")),
        wecom_enabled=bool(n.get("wecom_enabled", False)),
        wecom_webhook_url=str(n.get("wecom_webhook_url", "")),
    )

    config = PortfolioConfig(
        name=name,
        base_currency=base_currency,
        total_value=total_value,
        funds=funds,
        analysis=analysis,
        output=output,
        notify=notify,
    )

    validate_config(config)
    return config


def validate_config(config: PortfolioConfig) -> None:
    """Validate config integrity.

    Raises ValueError if weights don't sum to ~1.0 or fund types are invalid.
    """
    valid_types = {"a_share_mf", "a_share_etf", "overseas_etf"}

    for f in config.funds:
        if f.type not in valid_types:
            raise ValueError(
                f"Invalid fund type '{f.type}' for {f.code}. "
                f"Must be one of: {valid_types}"
            )
        if f.weight < 0 or f.weight > 1 or f.weight != f.weight:
            raise ValueError(
                f"Invalid weight {f.weight} for {f.code}. Must be a number between 0 and 1."
            )

    total_weight = sum(f.weight for f in config.funds)
    if abs(total_weight - 1.0) > 0.02:
        raise ValueError(
            f"Fund weights sum to {total_weight:.4f}, expected ~1.0. "
            f"Please adjust weights to sum to 1.0."
        )

    if not config.funds:
        raise ValueError("No funds configured. Please add at least one fund.")
