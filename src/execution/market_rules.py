"""
Shared market-resolution rules for signal generation, settlement, and exits.
"""

import re
from dataclasses import dataclass
from typing import Optional

from src.exchange.gamma_api import MarketInfo


@dataclass(frozen=True)
class MarketResolutionRule:
    """How a binary BTC market should resolve."""
    resolution_type: str  # move | above | below
    reference_price: Optional[float] = None


def derive_market_resolution_rule(market: MarketInfo) -> MarketResolutionRule:
    """
    Infer how the market settles from its question/slug.

    - `move`: classic short-horizon up/down market settled from start vs end price
    - `above`: yes wins when the end price is above a parsed strike
    - `below`: yes wins when the end price is below a parsed strike
    """
    searchable_text = " ".join(
        [
            str(market.question or ""),
            str(market.slug or "").replace("-", " "),
        ]
    ).lower()
    match = re.search(
        r"\b(above|below)\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b",
        searchable_text,
    )
    if not match:
        return MarketResolutionRule("move")

    direction = match.group(1)
    raw_price = match.group(2).replace(",", "")
    try:
        reference_price = float(raw_price)
    except ValueError:
        return MarketResolutionRule("move")

    return MarketResolutionRule(direction, reference_price)


def settles_yes(
    rule: MarketResolutionRule,
    end_price: float,
    start_price: Optional[float] = None,
) -> bool:
    """Return True when YES should resolve as the winner for a market rule."""
    if rule.resolution_type == "above":
        if rule.reference_price is None:
            raise ValueError("above-rule requires a reference price")
        return end_price > rule.reference_price

    if rule.resolution_type == "below":
        if rule.reference_price is None:
            raise ValueError("below-rule requires a reference price")
        return end_price < rule.reference_price

    if start_price is None:
        raise ValueError("move-rule requires a start price")
    return end_price > start_price


def is_position_favorable(
    rule: MarketResolutionRule,
    side: str,
    spot_price: float,
) -> Optional[bool]:
    """
    Return whether the current spot is on the favorable side of the rule.

    Returns None when the rule does not define a strike-based favorable side.
    """
    if rule.reference_price is None:
        return None

    yes_favorable: bool
    if rule.resolution_type == "above":
        yes_favorable = spot_price >= rule.reference_price
    elif rule.resolution_type == "below":
        yes_favorable = spot_price <= rule.reference_price
    else:
        return None

    if side == "BUY_YES":
        return yes_favorable
    if side == "BUY_NO":
        return not yes_favorable
    return None
