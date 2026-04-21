"""
Quantitative tool registry for tool-assisted agent evaluation.

Tools are made available to LLMs during S1/S2/S3 evaluation when --use-tools is set.
Each tool is described by a ToolSpec and implemented as a plain Python function.

Available built-in tools:
  - calculator    : evaluate arithmetic/math expressions
  - correlation   : Pearson correlation between two return series
  - volatility    : annualized volatility of a return series
  - mean_return   : annualized mean return of a series

Optional tool (requires SERPER_API_KEY):
  - web_search    : search the web for recent financial news/data
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    """Specification for a single quantitative tool."""
    name: str
    description: str
    input_schema: dict  # JSON Schema for the tool's input
    fn: Callable        # Python implementation


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
        raise ValueError(f"Could not evaluate expression {expression!r}: {exc}") from exc


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
    lines = [f"{i+1}. {r.get('title', '')}: {r.get('snippet', '')}" for i, r in enumerate(results)]
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


def get_tools(include_web_search: bool = False) -> list[ToolSpec]:
    """Return the list of available tools for agent evaluation."""
    tools = list(BUILTIN_TOOLS)
    if include_web_search:
        tools.append(_WEB_SEARCH_TOOL)
    return tools


def dispatch_tool(name: str, args: dict[str, Any], tools: list[ToolSpec]) -> Any:
    """Find a tool by name and call it with the given arguments."""
    for tool in tools:
        if tool.name == name:
            return tool.fn(**args)
    raise ValueError(f"Unknown tool: {name!r}. Available: {[t.name for t in tools]}")
