"""
Account-aware validation preflight for the Polymarket BTC bot.

This script checks model availability, wallet credentials, Polymarket account
state, market discovery, and whether a configured micro-budget can satisfy the
current venue minimum size. It never places or cancels orders.
"""

import argparse
import os
import sys
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import POLYMARKET, TRADING
from src.exchange.binance_rest import BinanceRESTClient
from src.exchange.gamma_api import GammaAPIClient, MarketInfo
from src.exchange.polymarket_client import PolymarketClient
from src.execution.inference import ModelInference
from src.utils.logging_config import setup_logging


def _mask(value: Optional[str], keep: int = 6) -> str:
    if not value:
        return "missing"
    value = str(value).strip()
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"


def _effective_budget() -> Optional[float]:
    budgets = [
        TRADING.max_order_notional if TRADING.max_order_notional > 0 else None,
        TRADING.order_notional if getattr(TRADING, "order_notional", 0.0) > 0 else None,
    ]
    for budget in budgets:
        if budget is not None and budget > 0:
            return budget
    return None


def _format_money(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:.4f}"


def _min_notional(price: Optional[float], min_order_size: Optional[float]) -> Optional[float]:
    if price is None or price <= 0 or min_order_size is None or min_order_size <= 0:
        return None
    return price * min_order_size


def _has_private_trading_creds() -> bool:
    return bool(
        POLYMARKET.private_key
        and POLYMARKET.api_key
        and POLYMARKET.api_secret
        and POLYMARKET.api_passphrase
    )


def _signature_type_diagnostics(current_collateral) -> list[dict]:
    """Probe alternate signature types to catch wallet-config mismatches."""
    if not _has_private_trading_creds():
        return []

    current_balance = current_collateral.balance if current_collateral else 0.0
    current_available = (
        current_collateral.available_to_trade if current_collateral else 0.0
    )

    findings: list[dict] = []
    for signature_type in (0, 1, 2):
        if signature_type == POLYMARKET.signature_type:
            continue

        probe_client = PolymarketClient(signature_type=signature_type)
        status = probe_client.get_collateral_balance_allowance()
        if status is None:
            continue

        if (
            status.available_to_trade > current_available + 1e-9
            or status.balance > current_balance + 1e-9
        ):
            findings.append(
                {
                    "signature_type": signature_type,
                    "balance": status.balance,
                    "allowance": status.allowance,
                    "available_to_trade": status.available_to_trade,
                }
            )

    findings.sort(
        key=lambda item: (
            item["available_to_trade"],
            item["balance"],
            item["allowance"],
        ),
        reverse=True,
    )
    return findings


def _print_section(title: str):
    print(f"\n[{title}]")


def main():
    parser = argparse.ArgumentParser(description="Run account-aware validation preflight")
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Override the per-order dollar budget used for venue-minimum checks",
    )
    args = parser.parse_args()

    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_output=False,
    )

    model = ModelInference()
    pm_client = PolymarketClient()
    gamma = GammaAPIClient()
    binance_rest = BinanceRESTClient()

    validation_blockers: list[str] = []
    live_blockers: list[str] = []
    warnings: list[str] = []

    try:
        _print_section("Config")
        print(f"Tracking address: {_mask(pm_client.tracking_address)}")
        print(f"Private key configured: {'yes' if bool(POLYMARKET.private_key) else 'no'}")
        print(f"CLOB API credentials configured: {'yes' if bool(POLYMARKET.api_key and POLYMARKET.api_secret and POLYMARKET.api_passphrase) else 'no'}")
        print(f"Trading access available: {'yes' if pm_client.has_trading_access else 'no'}")
        print(f"Validation-only env default: {bool(getattr(TRADING, 'validation_only_mode', False))}")
        print(
            "Cross-horizon live override: "
            f"{bool(getattr(TRADING, 'allow_non_5m_live_markets', False))}"
        )

        if not pm_client.has_trading_access:
            validation_blockers.append("Polymarket authenticated trading access is unavailable")
            live_blockers.append("Polymarket authenticated trading access is unavailable")

        _print_section("Model")
        model_loaded = model.load()
        print(f"Model load: {'ok' if model_loaded else 'failed'}")
        print(f"Model target horizon minutes: {model.target_horizon_minutes}")
        if not model_loaded:
            validation_blockers.append("Trained LightGBM model could not be loaded")
            live_blockers.append("Trained LightGBM model could not be loaded")

        _print_section("Binance Warmup")
        try:
            bars = binance_rest.fetch_recent_1m_klines(limit=120)
            print(f"Recent closed bars fetched: {len(bars)}")
            if bars.empty:
                validation_blockers.append("Binance warmup returned no closed minute bars")
                live_blockers.append("Binance warmup returned no closed minute bars")
        except Exception as exc:
            print(f"Binance warmup failed: {exc}")
            validation_blockers.append("Binance REST warmup failed")
            live_blockers.append("Binance REST warmup failed")

        _print_section("Polymarket Account")
        collateral = pm_client.get_collateral_balance_allowance()
        open_orders = pm_client.get_open_orders()
        if collateral is None:
            print("Collateral status: unavailable")
            validation_blockers.append("Collateral balance/allowance could not be read")
            live_blockers.append("Collateral balance/allowance could not be read")
        else:
            print(f"Balance: {_format_money(collateral.balance)}")
            print(f"Allowance: {_format_money(collateral.allowance)}")
            print(f"Available to trade: {_format_money(collateral.available_to_trade)}")
            if collateral.available_to_trade <= 0:
                live_blockers.append("No spendable collateral is available")
            if collateral.allowances_by_spender:
                print(
                    "Allowance spenders: "
                    f"{len(collateral.allowances_by_spender)} contracts tracked"
                )
        print(f"Open orders: {len(open_orders)}")
        if open_orders:
            warnings.append(f"Account has {len(open_orders)} open orders before validation run")

        signature_findings = _signature_type_diagnostics(collateral)
        if signature_findings:
            print("Alternate signature types with better account state:")
            for finding in signature_findings:
                print(
                    f"- sig={finding['signature_type']} "
                    f"balance={_format_money(finding['balance'])} "
                    f"available={_format_money(finding['available_to_trade'])}"
                )

            best_signature = signature_findings[0]
            signature_message = (
                f"Configured SIGNATURE_TYPE={POLYMARKET.signature_type} may be wrong "
                f"for this wallet; signature_type={best_signature['signature_type']} "
                f"sees balance={_format_money(best_signature['balance'])} "
                f"available={_format_money(best_signature['available_to_trade'])}"
            )
            live_blockers.append(signature_message)
            warnings.append(signature_message)

        _print_section("Market Discovery")
        market: Optional[MarketInfo] = gamma.get_active_btc_5m_market(force_refresh=True)
        if market is None:
            print("Active BTC market: not found")
            validation_blockers.append("Gamma market discovery did not return an active BTC market")
            live_blockers.append("Gamma market discovery did not return an active BTC market")
            btc_candidates = gamma.get_active_btc_market_candidates(limit=3)
            if btc_candidates:
                print("Other active BTC markets seen (not matched by the short-dated strategy):")
                for candidate in btc_candidates:
                    print(
                        f"- {candidate['slug']} | "
                        f"end={candidate['end_date'] or 'N/A'} | "
                        f"best_bid={candidate['best_bid'] or 'N/A'} | "
                        f"best_ask={candidate['best_ask'] or 'N/A'}"
                    )
                warnings.append(
                    "Active BTC markets exist, but none match the bot's short-dated discovery rules"
                )
            yes_ask = None
            no_ask = None
        else:
            print(f"Market slug: {market.slug}")
            print(f"End date: {market.end_date}")
            print(
                "Market interval minutes: "
                f"{market.market_interval_minutes if market.market_interval_minutes is not None else 'unknown'}"
            )
            print(f"Min order size: {market.min_order_size if market.min_order_size is not None else 'N/A'}")
            _, yes_ask = pm_client.get_best_bid_ask(market.yes_token_id)
            _, no_ask = pm_client.get_best_bid_ask(market.no_token_id)
            print(f"YES best ask: {yes_ask if yes_ask is not None else 'N/A'}")
            print(f"NO best ask: {no_ask if no_ask is not None else 'N/A'}")

            if (
                market.market_interval_minutes is not None
                and market.market_interval_minutes != model.target_horizon_minutes
            ):
                interval_warning = (
                    "Live discovery found a "
                    f"{market.market_interval_minutes}-minute BTC market, but the "
                    f"loaded model targets {model.target_horizon_minutes} minutes"
                )
                if getattr(TRADING, "allow_non_5m_live_markets", False):
                    warnings.append(interval_warning)
                else:
                    live_blockers.append(
                        "Discovered market interval does not match the loaded model "
                        "horizon; live trading is blocked"
                    )
                    warnings.append(interval_warning)

            yes_min_notional = _min_notional(yes_ask, market.min_order_size)
            no_min_notional = _min_notional(no_ask, market.min_order_size)
            print(f"YES min notional: {_format_money(yes_min_notional)}")
            print(f"NO min notional: {_format_money(no_min_notional)}")

            budget = args.budget if args.budget is not None else _effective_budget()
            print(f"Configured budget cap: {_format_money(budget)}")
            if budget is not None:
                cheapest_side = min(
                    value for value in [yes_min_notional, no_min_notional] if value is not None
                ) if any(value is not None for value in [yes_min_notional, no_min_notional]) else None
                if cheapest_side is not None and budget + 1e-9 < cheapest_side:
                    live_blockers.append(
                        "Current market minimum notional exceeds the configured per-order budget"
                    )

        _print_section("Recommended Command")
        print("venv/bin/python -m src.execution.engine --validation-only")

        _print_section("Verdict")
        if validation_blockers:
            print("Status: BLOCKED")
            for blocker in validation_blockers:
                print(f"- {blocker}")
        else:
            print("Status: READY_FOR_VALIDATION_ONLY")
            print("- Authenticated checks look healthy enough for a read-only validation run")

        if live_blockers:
            print("Live blockers:")
            for blocker in live_blockers:
                print(f"- {blocker}")

        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"- {warning}")

        raise SystemExit(1 if validation_blockers or live_blockers else 0)
    finally:
        binance_rest.close()
        gamma.close()


if __name__ == "__main__":
    main()
