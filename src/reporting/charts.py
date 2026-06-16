"""Chart generation using Plotly.

Each function returns a plotly Figure. All charts use a consistent color scheme.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
FUND_COLORS = {}  # populated on first use, maps fund_code -> color

_BENCHMARK_COLOR = "#000000"  # black for benchmarks
_MAX_SHARPE_COLOR = "#FF0000"  # red for max Sharpe point


def _get_color(fund_code: str) -> str:
    """Get consistent color for a fund code."""
    if fund_code not in FUND_COLORS:
        idx = len(FUND_COLORS) % len(COLORS)
        FUND_COLORS[fund_code] = COLORS[idx]
    return FUND_COLORS[fund_code]


def _str_path(path: Path | None) -> str | None:
    return str(path) if path else None


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_nav_series(
    nav_df: pd.DataFrame,
    fund_names: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Plot all fund NAV series normalized to 1.0 at start.

    Args:
        nav_df: Wide-format NAV DataFrame (index=date, columns=fund_code).
        fund_names: Dict mapping fund_code -> display name.
        output_path: If provided, save image to this path.

    Returns:
        Plotly Figure.
    """
    normalized = nav_df / nav_df.iloc[0]
    fig = go.Figure()

    for col in normalized.columns:
        name = fund_names.get(col, col) if fund_names else col
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col],
            mode="lines",
            name=name,
            line=dict(color=_get_color(col), width=1.5),
        ))

    fig.update_layout(
        title="① 净值走势 — 线越高=赚越多，所有线同涨同跌=没分散",
        xaxis_title="日期",
        yaxis_title="归一化净值 (起点=1.0)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        height=500,
        margin=dict(b=80),
    )
    fig.add_annotation(x=0.5, y=-0.15, xref="paper", yref="paper",
        text="怎么看：每根线是一只基金从买入到今天的走势。线越高说明赚得越多。如果11根线几乎一起上一起下，说明你的分散没做好——涨的时候开心，跌的时候一起亏。",
        showarrow=False, font=dict(size=9, color="#888"))

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_drawdown(
    daily_returns: pd.DataFrame,
    fund_names: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Plot drawdown over time for all funds.

    Args:
        daily_returns: DataFrame index=date, columns=fund_code.
        fund_names: Dict mapping fund_code -> display name.
        output_path: If provided, save to this path.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    for col in daily_returns.columns:
        cum = (1 + daily_returns[col].dropna()).cumprod()
        cummax = cum.cummax()
        drawdown = (cum - cummax) / cummax
        name = fund_names.get(col, col) if fund_names else col

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,  # percentage
            mode="lines",
            name=name,
            line=dict(color=_get_color(col), width=1),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(map(str, _color_to_rgb(_get_color(col))))},0.1)",
        ))

    fig.update_layout(
        title="② 回撤图 — 线越低=跌得越痛，看谁抗跌",
        xaxis_title="日期",
        yaxis_title="回撤幅度",
        yaxis=dict(ticksuffix="%"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        height=450,
        margin=dict(b=80),
    )
    fig.add_annotation(x=0.5, y=-0.15, xref="paper", yref="paper",
        text="怎么看：每条线的最低点=这只基金历史上最大亏了多少。线越往下说明跌的时候越惨。如果一只基金涨得多但回撤浅（比如跌不到10%），说明性价比高。",
        showarrow=False, font=dict(size=9, color="#888"))

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    fund_names: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Interactive correlation heatmap with annotated values.

    Args:
        corr_df: Square correlation DataFrame.
        fund_names: Dict mapping fund_code -> display name.
        output_path: If provided, save to this path.

    Returns:
        Plotly Figure.
    """
    labels = [fund_names.get(c, c) if fund_names else c for c in corr_df.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=labels,
        y=labels,
        zmin=-1, zmax=1,
        colorscale="RdBu_r",
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        title="③ 相关性热力图 — 红色越多=基金越像，越需要精简",
        template="plotly_white",
        height=500,
        width=600,
        margin=dict(b=80),
    )
    fig.add_annotation(x=0.5, y=-0.18, xref="paper", yref="paper",
        text="怎么看：每个格子是两只基金的相似度。红色越深=走势越像。全是红色=你买了一堆差不多的基金，白交多份管理费。理想情况应该红绿相间。",
        showarrow=False, font=dict(size=9, color="#888"))

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_dendrogram(
    linkage_matrix,
    labels: list[str],
    fund_names: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Plot hierarchical clustering dendrogram.

    Args:
        linkage_matrix: scipy linkage result.
        labels: Fund codes in original order.
        fund_names: Dict mapping fund_code -> display name.
        output_path: If provided, save to this path.

    Returns:
        Plotly Figure.
    """
    from scipy.cluster.hierarchy import dendrogram

    display_labels = [fund_names.get(l, l) if fund_names else l for l in labels]

    # Use matplotlib to compute dendrogram coordinates, then plot with plotly
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    dn = dendrogram(
        linkage_matrix,
        labels=display_labels,
        orientation="top",
        no_plot=True,
    )

    fig = go.Figure()

    # Plot each branch
    for xs, ys in zip(dn["icoord"], dn["dcoord"]):
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#555555", width=1.5),
            showlegend=False,
            hoverinfo="none",
        ))

    # Leaf labels
    leaf_colors = {}
    if fund_names:
        # Map back to original codes for color
        reverse_map = {v: k for k, v in fund_names.items()}
        for label in dn["ivl"]:
            code = reverse_map.get(label, label)
            leaf_colors[label] = _get_color(code)

    for i, (label, x) in enumerate(zip(dn["ivl"], dn["leaves"])):
        color = leaf_colors.get(label, "#333333")
        fig.add_annotation(
            x=x,
            y=0,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="top",
            textangle=-90,
            font=dict(color=color, size=10),
        )

    fig.update_layout(
        title="④ 聚类树 — 连在一起的基金走势像，应该只留一个",
        xaxis=dict(showticklabels=False),
        yaxis_title="相似距离（越近越像）",
        template="plotly_white",
        height=420,
        showlegend=False,
        margin=dict(b=80),
    )
    fig.add_annotation(x=0.5, y=-0.22, xref="paper", yref="paper",
        text="怎么看：竖线连在一起的基金=走势高度相似。比如「半导体A和C」连在一起，说明它们其实是一只基金，没必要两个都买。应该从每组里挑一个。",
        showarrow=False, font=dict(size=9, color="#888"))

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_efficient_frontier(
    frontier: dict,
    fund_returns: pd.DataFrame,
    fund_names: dict[str, str] | None = None,
    max_sharpe_weight: np.ndarray | None = None,
    risk_parity_weight: np.ndarray | None = None,
    min_var_weight: np.ndarray | None = None,
    exp_ret: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Plot efficient frontier with individual assets and special portfolios.

    Args:
        frontier: Output from optimization.efficient_frontier().
        fund_returns: Daily returns DataFrame.
        fund_names: Dict mapping fund_code -> display name.
        max_sharpe_weight: Max Sharpe portfolio weights.
        risk_parity_weight: Risk parity portfolio weights.
        min_var_weight: Min variance portfolio weights.
        exp_ret: Expected returns array.
        cov: Covariance matrix.
        output_path: If provided, save to this path.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Efficient frontier curve
    fig.add_trace(go.Scatter(
        x=frontier["volatilities"],
        y=frontier["returns"],
        mode="lines",
        name="有效前沿",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Max Sharpe on frontier
    sharpe_arr = np.array(frontier["sharpe_ratios"])
    max_idx = sharpe_arr.argmax()
    fig.add_trace(go.Scatter(
        x=[frontier["volatilities"][max_idx]],
        y=[frontier["returns"][max_idx]],
        mode="markers",
        name="最大夏普组合",
        marker=dict(color=_MAX_SHARPE_COLOR, size=14, symbol="star"),
        hovertemplate="Max Sharpe<br>Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Individual assets — use short names to avoid label overlap
    if exp_ret is not None and cov is not None:
        from src.analysis.optimization import portfolio_volatility
        ann_vols = [np.sqrt(cov[i, i]) for i in range(len(exp_ret))]
        codes = list(fund_returns.columns)
        for i, code in enumerate(codes):
            full_name = fund_names.get(code, code) if fund_names else code
            short = _short_name(full_name)
            fig.add_trace(go.Scatter(
                x=[ann_vols[i]],
                y=[exp_ret[i]],
                mode="markers",
                name=short,
                marker=dict(color=_get_color(code), size=9),
                hovertemplate=f"<b>{full_name}</b><br>波动: %{{x:.1%}}<br>收益: %{{y:.1%}}<extra></extra>",
            ))

    # Special portfolios
    if max_sharpe_weight is not None and exp_ret is not None and cov is not None:
        from src.analysis.optimization import portfolio_return, portfolio_volatility
        ms_vol = portfolio_volatility(max_sharpe_weight, cov)
        ms_ret = portfolio_return(max_sharpe_weight, exp_ret)
        fig.add_trace(go.Scatter(
            x=[ms_vol],
            y=[ms_ret],
            mode="markers",
            name="最大夏普 (当前)",
            marker=dict(color=_MAX_SHARPE_COLOR, size=12, symbol="x"),
        ))

    if risk_parity_weight is not None and exp_ret is not None and cov is not None:
        from src.analysis.optimization import portfolio_return, portfolio_volatility
        rp_vol = portfolio_volatility(risk_parity_weight, cov)
        rp_ret = portfolio_return(risk_parity_weight, exp_ret)
        fig.add_trace(go.Scatter(
            x=[rp_vol],
            y=[rp_ret],
            mode="markers",
            name="风险平价",
            marker=dict(color="#2ca02c", size=12, symbol="diamond"),
        ))

    if min_var_weight is not None and exp_ret is not None and cov is not None:
        from src.analysis.optimization import portfolio_return, portfolio_volatility
        mv_vol = portfolio_volatility(min_var_weight, cov)
        mv_ret = portfolio_return(min_var_weight, exp_ret)
        fig.add_trace(go.Scatter(
            x=[mv_vol],
            y=[mv_ret],
            mode="markers",
            name="最小方差",
            marker=dict(color="#9467bd", size=12, symbol="triangle-up"),
        ))

    fig.update_layout(
        title="⑤ 有效前沿 — 曲线上的点=最优组合，离曲线越远=越差",
        xaxis_title="年化波动率（风险）",
        yaxis_title="年化收益率",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
        height=550,
        margin=dict(b=100),
    )
    fig.add_annotation(x=0.5, y=-0.22, xref="paper", yref="paper",
        text="怎么看：每个散点=一只基金，蓝色曲线=往右上角越好的最优组合。点离曲线越远=这只基金单独持有性价比越低。红星=计算机算出的最佳搭配。散点集中在左上方=收益高风险低才是好基金。",
        showarrow=False, font=dict(size=9, color="#888"))

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_risk_contribution(
    weights: np.ndarray,
    cov: np.ndarray,
    fund_codes: list[str],
    fund_names: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Pie chart of risk contributions.

    RC_i = w_i * (Cov @ w)_i / portfolio_vol
    """
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal = cov @ weights
    risk_contrib = weights * marginal / port_vol

    labels = [fund_names.get(c, c) if fund_names else c for c in fund_codes]
    colors = [_get_color(c) for c in fund_codes]

    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=risk_contrib,
        marker=dict(colors=colors),
        texttemplate="%{label}<br>%{percent:.1%}",
        hole=0.3,
    ))

    fig.update_layout(
        title="⑥ 风险贡献饼图 — 饼越大=这只基金拖累你越多",
        template="plotly_white",
        height=480,
        margin=dict(b=80),
    )
    fig.add_annotation(x=0.5, y=-0.15, xref="paper", yref="paper",
        text="怎么看：每块饼=这只基金在组合总风险里的占比。理想情况是每块差不多大。如果某一块特别大，说明你的风险被少数基金绑架了。",
        showarrow=False, font=dict(size=9, color="#888"))

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_rolling_metrics(
    rolling_data: pd.DataFrame,
    title: str = "Rolling Metrics",
    output_path: Path | None = None,
) -> go.Figure:
    """Plot rolling alpha, beta, or other rolling metrics.

    Args:
        rolling_data: DataFrame index=date, columns=metric_name.
        title: Chart title.
        output_path: If provided, save to this path.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    for col in rolling_data.columns:
        fig.add_trace(go.Scatter(
            x=rolling_data.index,
            y=rolling_data[col],
            mode="lines",
            name=col,
            line=dict(width=1.5),
        ))

    # Add zero line
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=0.5))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        height=400,
    )

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_excess_return_curve(
    excess_curves: dict[str, pd.Series],
    title: str = "Cumulative Excess Returns vs Benchmark",
    output_path: Path | None = None,
) -> go.Figure:
    """Plot cumulative excess return curves for multiple funds.

    Args:
        excess_curves: Dict mapping label -> Series of cumulative excess returns.
        title: Chart title.
        output_path: If provided, save to this path.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    for i, (label, curve) in enumerate(excess_curves.items()):
        fig.add_trace(go.Scatter(
            x=curve.index,
            y=curve - 1,  # convert to % change
            mode="lines",
            name=label,
            line=dict(width=1.5),
        ))

    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=0.5))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Excess Return",
        yaxis=dict(tickformat=".0%"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        height=400,
    )

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


def plot_portfolio_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    benchmark_name: str = "Benchmark",
    output_path: Path | None = None,
) -> go.Figure:
    """Plot portfolio cumulative return vs benchmark.

    Args:
        portfolio_returns: Portfolio daily returns.
        benchmark_returns: Benchmark daily returns (optional).
        benchmark_name: Name for the benchmark.
        output_path: If provided, save to this path.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    cum_port = (1 + portfolio_returns.dropna()).cumprod()
    fig.add_trace(go.Scatter(
        x=cum_port.index,
        y=cum_port,
        mode="lines",
        name="Portfolio",
        line=dict(color="#1f77b4", width=2),
    ))

    if benchmark_returns is not None:
        cum_bm = (1 + benchmark_returns.dropna()).cumprod()
        # Align dates
        common_idx = cum_port.index.intersection(cum_bm.index)
        fig.add_trace(go.Scatter(
            x=common_idx,
            y=cum_bm[common_idx],
            mode="lines",
            name=benchmark_name,
            line=dict(color=_BENCHMARK_COLOR, width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title="⑦ 组合整体表现 — 你所有钱放一起的总收益曲线",
        xaxis_title="日期",
        yaxis_title="累计收益（起点=1.0）",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
        height=470,
        margin=dict(b=80),
    )
    fig.add_annotation(x=0.5, y=-0.15, xref="paper", yref="paper",
        text="怎么看：这跟线就是你按当前比例持有所有基金的总盈亏。如果线往下走=整体在亏钱。核心问题是，线波动大不大、回撤深不深。",
        showarrow=False, font=dict(size=9, color="#888"))

    if output_path:
        fig.write_image(_str_path(output_path), scale=2)
    return fig


# ---------------------------------------------------------------------------
# Batch chart generation
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to (r, g, b) tuple (legacy)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _color_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse color string in rgb() or hex format to (r, g, b) tuple."""
    if color.startswith("rgb"):
        import re
        nums = re.findall(r"\d+", color)
        return tuple(int(n) for n in nums[:3])
    elif color.startswith("#"):
        return _hex_to_rgb(color)
    else:
        return (100, 100, 100)


def _short_name(name: str, max_len: int = 10) -> str:
    """Truncate fund name for display on charts."""
    # Remove common suffixes
    for suffix in ["混合发起", "股票发起式", "混合", "联接"]:
        name = name.replace(suffix, "")
    # Keep key identifier
    if len(name) > max_len:
        # Try to keep company + keyword
        parts = name.replace("(", " ").replace(")", "").replace("（", " ").replace("）", "").split()
        if len(parts) >= 2:
            name = parts[0][:4] + parts[-1][:6]
        else:
            name = name[:max_len]
    return name


def save_all_charts(
    nav_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    corr_df: pd.DataFrame,
    linkage_result,
    labels: list[str],
    frontier: dict,
    max_sharpe_weights: np.ndarray,
    risk_parity_weights: np.ndarray,
    min_var_weights: np.ndarray,
    exp_ret: np.ndarray,
    cov: np.ndarray,
    portfolio_returns: pd.Series,
    fund_names: dict[str, str],
    output_dir: str | Path,
    chart_format: str = "png",
    nav_display: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
    benchmark_name: str | None = None,
) -> list[Path]:
    """Generate all charts and save to output directory.

    Returns list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    charts_to_generate = []  # all chart sections removed from report

    for name, chart_fn in charts_to_generate:
        try:
            path = output_dir / f"{name}.{chart_format}"
            fig = chart_fn()
            if chart_format == "png":
                fig.write_image(str(path), scale=2, engine="kaleido")
            elif chart_format == "html":
                fig.write_html(str(path))
            else:
                logger.warning("Unsupported chart format '%s' for %s, saving as png", chart_format, name)
                fig.write_image(str(path.with_suffix('.png')), scale=2, engine="kaleido")
                path = path.with_suffix('.png')
            saved.append(path)
            logger.info("Saved chart: %s", path)
        except Exception as e:
            logger.error("Failed to generate chart %s: %s", name, e)

    return saved
