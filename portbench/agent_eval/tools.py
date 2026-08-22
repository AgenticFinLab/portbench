"""
Quantitative tool registry for tool-assisted agent evaluation.

Tools are made available to LLMs during S1/S2/S3 evaluation when --use-tools is set.
Each tool is described by a ToolSpec and implemented as a plain Python function.

Available built-in tools:
  - calculator    : evaluate arithmetic/math expressions
  - correlation   : Pearson correlation between two return series
  - volatility    : annualized volatility of a return series
  - mean_return   : annualized mean return of a series

Snapshot-bound tools:
  - portfolio_risk    : lookback volatility, VaR/CVaR, and drawdown
  - risk_contribution : lookback variance contribution by asset
  - execution_cost    : turnover and deterministic transaction-cost estimate

Optional tool (requires SERPER_API_KEY):
  - web_search    : search the web for recent financial news/data
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class ToolSpec:
    """Specification for a single quantitative tool."""

    name: str
    description: str
    input_schema: dict  # JSON Schema for the tool's input
    fn: Callable  # Python implementation


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _calc(expression: str) -> float:
    """Safely evaluate a math expression using a restricted namespace."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["abs"] = abs
    allowed["round"] = round
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return float(result)
    except Exception as exc:
        raise ValueError(
            f"Could not evaluate expression {expression!r}: {exc}"
        ) from exc


def _pearson_correlation(a: list[float], b: list[float]) -> float:
    """Compute Pearson correlation coefficient between two return series."""
    if len(a) != len(b):
        raise ValueError(f"Series lengths differ: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 observations to compute correlation.")
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    std_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if std_a == 0 or std_b == 0:
        return 0.0
    return cov / (std_a * std_b)


def _volatility(returns: list[float], annualize: bool = True) -> float:
    """Compute standard deviation of returns, optionally annualized (252 trading days)."""
    n = len(returns)
    if n < 2:
        raise ValueError("Need at least 2 observations to compute volatility.")
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    vol = math.sqrt(variance)
    return vol * math.sqrt(252) if annualize else vol


def _mean_return(returns: list[float], annualize: bool = True) -> float:
    """Compute mean return, optionally annualized (252 trading days)."""
    if not returns:
        raise ValueError("Empty returns list.")
    daily_mean = sum(returns) / len(returns)
    return daily_mean * 252 if annualize else daily_mean


def _weight_vector(snapshot: Any, weights: dict[str, float]) -> tuple[pd.DataFrame, np.ndarray]:
    """Align declared weights to historical return columns."""
    # Build the universe from lookback returns so forward-only assets cannot enter the tool.
    frame = pd.DataFrame(snapshot.return_data).dropna(how="all").fillna(0.0)
    if frame.empty:
        raise ValueError("Historical return data is required.")
    # Clip negative declarations because every evaluated portfolio is long-only.
    vector = np.array(
        [max(0.0, float(weights.get(asset, 0.0))) for asset in frame.columns]
    )
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("At least one positive portfolio weight is required.")
    return frame, vector / total


def _portfolio_risk(snapshot: Any, weights: dict[str, float]) -> dict[str, float]:
    """Compute lookback-only volatility, historical VaR/CVaR, and drawdown."""
    frame, vector = _weight_vector(snapshot, weights)
    # Convert aligned asset returns into one historical portfolio return series.
    returns = frame.to_numpy(dtype=float) @ vector
    if len(returns) < 2:
        raise ValueError("At least two historical observations are required.")

    # Derive drawdown from compounded wealth rather than summing returns.
    wealth = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / np.maximum(running_peak, 1e-12) - 1.0
    # Use the empirical lower tail to avoid a distributional VaR assumption.
    quantile = float(np.quantile(returns, 0.05))
    tail = returns[returns <= quantile]
    return {
        "annualized_volatility": float(np.std(returns, ddof=1) * math.sqrt(252)),
        "historical_var_95": float(max(0.0, -quantile)),
        "historical_cvar_95": float(max(0.0, -float(np.mean(tail)))) if len(tail) else 0.0,
        "max_drawdown": float(np.min(drawdowns)) if len(drawdowns) else 0.0,
    }


def _risk_contribution(snapshot: Any, weights: dict[str, float]) -> dict[str, float]:
    """Compute each asset's share of lookback portfolio variance."""
    frame, vector = _weight_vector(snapshot, weights)
    # Annualize the lookback covariance before computing marginal contributions.
    covariance = frame.cov().fillna(0.0).to_numpy(dtype=float) * 252.0
    marginal = covariance @ vector
    total_variance = float(vector @ marginal)
    if total_variance <= 0.0:
        return {str(asset): 0.0 for asset in frame.columns}
    # Divide component variance by total variance to return contribution shares.
    return {
        str(asset): float(vector[index] * marginal[index] / total_variance)
        for index, asset in enumerate(frame.columns)
    }


def _execution_cost(
    snapshot: Any,
    target_weights: dict[str, float],
    slippage_rate: float = 0.001,
    commission_rate: float = 0.0005,
) -> dict[str, float]:
    """Estimate turnover and deterministic trading cost from current weights."""
    # Include assets that disappear from or enter the target portfolio.
    assets = set(snapshot.current_weights) | set(target_weights)
    turnover = sum(
        abs(float(target_weights.get(asset, 0.0)) - float(snapshot.current_weights.get(asset, 0.0)))
        for asset in assets
    )
    traded_notional = float(snapshot.portfolio_value) * turnover
    # Apply linear slippage and commission to the same traded notional.
    return {
        "turnover": float(turnover),
        "traded_notional": traded_notional,
        "estimated_cost": traded_notional * (float(slippage_rate) + float(commission_rate)),
    }


def _web_search(query: str, n_results: int = 3) -> str:
    """
    Search the web for financial information using the Serper API.
    Requires SERPER_API_KEY in environment.
    """
    import json
    import urllib.request

    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return "Web search unavailable: SERPER_API_KEY not set in environment."

    payload = json.dumps({"q": query, "num": n_results}).encode()
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    results = data.get("organic", [])[:n_results]
    if not results:
        return "No search results found."
    lines = [
        f"{i+1}. {r.get('title', '')}: {r.get('snippet', '')}"
        for i, r in enumerate(results)
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="calculator",
        description=(
            "Evaluate a mathematical expression. Supports standard arithmetic, "
            "math functions (sqrt, log, exp, sin, cos, etc.), and abs/round. "
            "Use this to compute returns, Sharpe ratios, position sizes, etc."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '(0.12 - 0.02) / 0.15'",
                },
            },
            "required": ["expression"],
        },
        fn=lambda expression: _calc(expression),
    ),
    ToolSpec(
        name="correlation",
        description=(
            "Compute the Pearson correlation coefficient between two return series. "
            "Returns a float in [-1, 1]. Use to measure diversification between assets."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "First return series (daily returns as decimals).",
                },
                "b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Second return series (daily returns as decimals).",
                },
            },
            "required": ["a", "b"],
        },
        fn=lambda a, b: _pearson_correlation(a, b),
    ),
    ToolSpec(
        name="volatility",
        description=(
            "Compute the annualized volatility (standard deviation) of a return series. "
            "Assumes daily returns; annualizes by multiplying by sqrt(252)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Daily return series as decimals.",
                },
                "annualize": {
                    "type": "boolean",
                    "description": "Whether to annualize (default: true).",
                },
            },
            "required": ["returns"],
        },
        fn=lambda returns, annualize=True: _volatility(returns, annualize),
    ),
    ToolSpec(
        name="mean_return",
        description=(
            "Compute the annualized mean return of a return series. "
            "Assumes daily returns; annualizes by multiplying by 252."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Daily return series as decimals.",
                },
                "annualize": {
                    "type": "boolean",
                    "description": "Whether to annualize (default: true).",
                },
            },
            "required": ["returns"],
        },
        fn=lambda returns, annualize=True: _mean_return(returns, annualize),
    ),
]

_WEB_SEARCH_TOOL = ToolSpec(
    name="web_search",
    description=(
        "Search the web for recent financial news, market data, or economic indicators. "
        "Returns top snippets from search results. Requires SERPER_API_KEY."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return (default: 3, max: 10).",
            },
        },
        "required": ["query"],
    },
    fn=lambda query, n_results=3: _web_search(query, n_results),
)


def _snapshot_tools(snapshot: Any) -> list[ToolSpec]:
    """Bind deterministic Point-in-Time portfolio tools to one snapshot."""
    # Reuse one schema so all three tools accept the same weight representation.
    weights_schema = {
        "type": "object",
        "additionalProperties": {"type": "number"},
        "description": "Portfolio weights keyed by asset symbol.",
    }
    return [
        ToolSpec(
            name="portfolio_risk",
            description="Compute lookback annualized volatility, historical VaR/CVaR, and max drawdown.",
            input_schema={
                "type": "object",
                "properties": {"weights": weights_schema},
                "required": ["weights"],
            },
            fn=lambda weights: _portfolio_risk(snapshot, weights),
        ),
        ToolSpec(
            name="risk_contribution",
            description="Compute lookback variance contribution by asset for candidate weights.",
            input_schema={
                "type": "object",
                "properties": {"weights": weights_schema},
                "required": ["weights"],
            },
            fn=lambda weights: _risk_contribution(snapshot, weights),
        ),
        ToolSpec(
            name="execution_cost",
            description="Estimate turnover and trading cost relative to current portfolio weights.",
            input_schema={
                "type": "object",
                "properties": {
                    "target_weights": weights_schema,
                    "slippage_rate": {"type": "number", "default": 0.001},
                    "commission_rate": {"type": "number", "default": 0.0005},
                },
                "required": ["target_weights"],
            },
            fn=lambda target_weights, slippage_rate=0.001, commission_rate=0.0005: _execution_cost(
                snapshot, target_weights, slippage_rate, commission_rate
            ),
        ),
    ]


def get_tools(include_web_search: bool = False, snapshot: Any = None) -> list[ToolSpec]:
    """Return the list of available tools for agent evaluation."""
    tools = list(BUILTIN_TOOLS)
    # Snapshot tools close over the current PiT state and never access forward returns.
    if snapshot is not None:
        tools.extend(_snapshot_tools(snapshot))
    if include_web_search:
        tools.append(_WEB_SEARCH_TOOL)
    return tools


def dispatch_tool(name: str, args: dict[str, Any], tools: list[ToolSpec]) -> Any:
    """Find a tool by name and call it with the given arguments."""
    for tool in tools:
        if tool.name == name:
            return tool.fn(**args)
    raise ValueError(f"Unknown tool: {name!r}. Available: {[t.name for t in tools]}")
