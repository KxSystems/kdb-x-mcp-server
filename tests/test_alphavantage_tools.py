#!/usr/bin/env python3
"""
AlphaVantage MCP Tools Test Suite

This script tests the AlphaVantage algo trading tools with real market data
using natural language-style test scenarios.
"""

import asyncio
import sys
import json
from datetime import datetime

# Add source path
sys.path.insert(0, '/home/user/kdb-x-mcp-server-tools/src/mcp_server/tools/alphavantage')

# Import data provider
from data_provider import (
    get_ohlcv, get_fx_ohlcv, get_quote, get_news_sentiment, get_fx_rate,
    get_rate_limit_status
)

# Import tool implementations
from moving_averages import av_sma_impl, av_ema_impl, av_ma_compare_impl
from momentum import av_rsi_impl, av_macd_impl, av_stochastic_impl
from trend import av_adx_impl, av_supertrend_impl
from volatility import av_bbands_impl, av_atr_impl
from volume import av_obv_impl, av_vwap_impl
from oscillators import av_cci_impl, av_willr_impl
from signals import av_golden_cross_impl, av_multi_signal_impl, av_trend_strength_impl
from sentiment import av_news_sentiment_impl, av_sentiment_trend_impl
from risk import av_position_size_impl, av_sharpe_impl, av_max_drawdown_impl
from fx import av_fx_quote_impl, av_fx_technical_impl, av_fx_pivot_impl


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")


def print_test(query, result):
    status = result.get('status', 'unknown')
    if status == 'success':
        print(f"\n{Colors.GREEN}[PASS]{Colors.RESET} {query}")
    else:
        print(f"\n{Colors.RED}[FAIL]{Colors.RESET} {query}")
        print(f"  Error: {result.get('message', 'Unknown error')}")
        return False
    return True


def print_result(key, value):
    if isinstance(value, float):
        print(f"  {Colors.YELLOW}{key}:{Colors.RESET} {value:.4f}")
    else:
        print(f"  {Colors.YELLOW}{key}:{Colors.RESET} {value}")


async def run_tests():
    """Run comprehensive tests on AlphaVantage tools."""

    print_header("AlphaVantage MCP Tools Test Suite")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    passed = 0
    failed = 0

    # =========================================================================
    # TEST 1: Moving Averages
    # =========================================================================
    print_header("1. Moving Average Tools")

    # Natural language query: "What is the 20-period SMA for Apple?"
    print("\nQuery: What is the 20-period SMA for Apple?")
    result = await av_sma_impl("AAPL", period=20, interval="60min")
    if print_test("av_sma(AAPL, period=20)", result):
        print_result("Current Price", result.get('current_price'))
        print_result("SMA(20)", result.get('current_sma'))
        print_result("Trend", result.get('trend'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "Show me EMA(12) and EMA(26) for Microsoft"
    print("\nQuery: Show me EMA(12) for Microsoft")
    result = await av_ema_impl("MSFT", period=12, interval="60min")
    if print_test("av_ema(MSFT, period=12)", result):
        print_result("Current Price", result.get('current_price'))
        print_result("EMA(12)", result.get('current_ema'))
        print_result("EMA Slope", result.get('ema_slope'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 2: Momentum Indicators
    # =========================================================================
    print_header("2. Momentum Indicator Tools")

    # Natural language query: "Is Tesla overbought or oversold? Check RSI"
    print("\nQuery: Is Tesla overbought or oversold? Check RSI")
    result = await av_rsi_impl("TSLA", period=14, interval="60min")
    if print_test("av_rsi(TSLA, period=14)", result):
        print_result("RSI", result.get('current_rsi'))
        print_result("Signal", result.get('signal'))
        print_result("Action", result.get('action'))
        print_result("Divergence", result.get('divergence'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "What's the MACD for Google? Any crossover signals?"
    print("\nQuery: What's the MACD for Google? Any crossover signals?")
    result = await av_macd_impl("GOOGL", fast_period=12, slow_period=26, signal_period=9)
    if print_test("av_macd(GOOGL)", result):
        print_result("MACD Line", result.get('current_macd'))
        print_result("Signal Line", result.get('current_signal'))
        print_result("Histogram", result.get('current_histogram'))
        print_result("Crossover", result.get('crossover_detected'))
        print_result("Trend", result.get('trend'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "Show Stochastic for Amazon"
    print("\nQuery: Show Stochastic oscillator for Amazon")
    result = await av_stochastic_impl("AMZN", k_period=14, d_period=3)
    if print_test("av_stochastic(AMZN)", result):
        print_result("%K", result.get('current_k'))
        print_result("%D", result.get('current_d'))
        print_result("Signal", result.get('signal'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 3: Trend Indicators
    # =========================================================================
    print_header("3. Trend Indicator Tools")

    # Natural language query: "How strong is the trend for Nvidia? Use ADX"
    print("\nQuery: How strong is the trend for Nvidia? Use ADX")
    result = await av_adx_impl("NVDA", period=14)
    if print_test("av_adx(NVDA)", result):
        print_result("ADX", result.get('current_adx'))
        print_result("+DI", result.get('current_plus_di'))
        print_result("-DI", result.get('current_minus_di'))
        print_result("Trend Strength", result.get('trend_strength'))
        print_result("Direction", result.get('trend_direction'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "What's the SuperTrend for Meta?"
    print("\nQuery: What's the SuperTrend signal for Meta?")
    result = await av_supertrend_impl("META", period=10, multiplier=3.0)
    if print_test("av_supertrend(META)", result):
        print_result("SuperTrend", result.get('supertrend'))
        print_result("Current Price", result.get('current_price'))
        print_result("Trend", result.get('trend'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 4: Volatility Indicators
    # =========================================================================
    print_header("4. Volatility Indicator Tools")

    # Natural language query: "Calculate Bollinger Bands for Apple"
    print("\nQuery: Calculate Bollinger Bands for Apple")
    result = await av_bbands_impl("AAPL", period=20, std_dev=2.0)
    if print_test("av_bbands(AAPL)", result):
        print_result("Upper Band", result.get('upper_band'))
        print_result("Middle Band", result.get('middle_band'))
        print_result("Lower Band", result.get('lower_band'))
        print_result("Bandwidth", result.get('bandwidth'))
        print_result("%B", result.get('percent_b'))
        print_result("Position", result.get('position'))
        print_result("Squeeze", result.get('squeeze_detected'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "What's the ATR for Tesla? I need it for stop-loss"
    print("\nQuery: What's the ATR for Tesla? I need it for stop-loss")
    result = await av_atr_impl("TSLA", period=14)
    if print_test("av_atr(TSLA)", result):
        print_result("ATR", result.get('current_atr'))
        print_result("Normalized ATR %", result.get('normalized_atr'))
        print_result("Volatility", result.get('volatility'))
        print_result("Suggested Stop Distance", result.get('suggested_stop_loss_distance'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 5: Volume Indicators
    # =========================================================================
    print_header("5. Volume Indicator Tools")

    # Natural language query: "What's the OBV showing for Microsoft?"
    print("\nQuery: What's the OBV showing for Microsoft?")
    result = await av_obv_impl("MSFT")
    if print_test("av_obv(MSFT)", result):
        print_result("OBV", result.get('current_obv'))
        print_result("OBV Trend", result.get('obv_trend'))
        print_result("Price Trend", result.get('price_trend'))
        print_result("Divergence", result.get('divergence'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "Calculate VWAP for Apple"
    print("\nQuery: Calculate VWAP for Apple")
    result = await av_vwap_impl("AAPL")
    if print_test("av_vwap(AAPL)", result):
        print_result("VWAP", result.get('current_vwap'))
        print_result("Current Price", result.get('current_price'))
        print_result("Price vs VWAP %", result.get('price_vs_vwap_pct'))
        print_result("Bias", result.get('bias'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 6: Oscillators
    # =========================================================================
    print_header("6. Oscillator Tools")

    # Natural language query: "What's the CCI for Amazon?"
    print("\nQuery: What's the CCI for Amazon?")
    result = await av_cci_impl("AMZN", period=20)
    if print_test("av_cci(AMZN)", result):
        print_result("CCI", result.get('current_cci'))
        print_result("Signal", result.get('signal'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "Check Williams %R for Google"
    print("\nQuery: Check Williams %R for Google")
    result = await av_willr_impl("GOOGL", period=14)
    if print_test("av_willr(GOOGL)", result):
        print_result("Williams %R", result.get('current_willr'))
        print_result("Signal", result.get('signal'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 7: Signal Detection
    # =========================================================================
    print_header("7. Signal Detection Tools")

    # Natural language query: "Is there a golden cross or death cross for Apple?"
    print("\nQuery: Is there a golden cross or death cross for Apple?")
    result = await av_golden_cross_impl("AAPL", fast_period=50, slow_period=200)
    if print_test("av_golden_cross(AAPL)", result):
        print_result("SMA(50)", result.get('sma_fast'))
        print_result("SMA(200)", result.get('sma_slow'))
        print_result("Crossover", result.get('crossover_detected'))
        print_result("Position", result.get('current_position'))
        print_result("Spread %", result.get('spread_percent'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "Give me a multi-indicator analysis for Tesla"
    print("\nQuery: Give me a multi-indicator analysis for Tesla")
    result = await av_multi_signal_impl("TSLA")
    if print_test("av_multi_signal(TSLA)", result):
        print_result("Overall Signal", result.get('overall_signal'))
        print_result("Bullish Signals", result.get('bullish_signals'))
        print_result("Bearish Signals", result.get('bearish_signals'))
        print_result("Confidence %", result.get('confidence'))
        print(f"  {Colors.YELLOW}Individual Signals:{Colors.RESET}")
        for signal_name, signal_value in result.get('signals', {}).items():
            print(f"    - {signal_name}: {signal_value}")
        passed += 1
    else:
        failed += 1

    # Natural language query: "Analyze trend strength for Nvidia"
    print("\nQuery: Analyze trend strength for Nvidia")
    result = await av_trend_strength_impl("NVDA")
    if print_test("av_trend_strength(NVDA)", result):
        print_result("ADX", result.get('adx'))
        print_result("Direction", result.get('trend_direction'))
        print_result("Strength", result.get('trend_strength'))
        print_result("Recommendation", result.get('recommended_action'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 8: Sentiment Analysis
    # =========================================================================
    print_header("8. Sentiment Analysis Tools")

    # Natural language query: "What's the news sentiment for Apple and Microsoft?"
    print("\nQuery: What's the news sentiment for Apple and Microsoft?")
    result = await av_news_sentiment_impl("AAPL,MSFT", limit=20)
    if print_test("av_news_sentiment(AAPL,MSFT)", result):
        print_result("Total Articles", result.get('total_articles'))
        print_result("Overall Sentiment", result.get('overall_sentiment'))
        print_result("Sentiment Score", result.get('overall_sentiment_score'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "How is sentiment trending for Tesla?"
    print("\nQuery: How is sentiment trending for Tesla?")
    result = await av_sentiment_trend_impl("TSLA", limit=50)
    if print_test("av_sentiment_trend(TSLA)", result):
        print_result("Current Sentiment", result.get('current_sentiment'))
        print_result("Avg Sentiment", result.get('avg_sentiment'))
        print_result("Sentiment Trend", result.get('sentiment_trend'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 9: Risk Management
    # =========================================================================
    print_header("9. Risk Management Tools")

    # Natural language query: "How many shares of Apple should I buy with $100K account risking 2%?"
    print("\nQuery: How many shares of Apple should I buy with $100K account risking 2%?")
    result = await av_position_size_impl("AAPL", account_size=100000, risk_percent=2.0)
    if print_test("av_position_size(AAPL, $100K, 2%)", result):
        print_result("Current Price", result.get('current_price'))
        print_result("Risk Amount", f"${result.get('risk_amount')}")
        print_result("Stop Price", result.get('stop_price'))
        print_result("Recommended Shares", result.get('recommended_shares'))
        print_result("Position Value", f"${result.get('position_value')}")
        passed += 1
    else:
        failed += 1

    # Natural language query: "What's the Sharpe ratio for Microsoft?"
    print("\nQuery: What's the Sharpe ratio for Microsoft?")
    result = await av_sharpe_impl("MSFT", risk_free_rate=0.05)
    if print_test("av_sharpe(MSFT)", result):
        print_result("Sharpe Ratio", result.get('sharpe_ratio'))
        print_result("Sortino Ratio", result.get('sortino_ratio'))
        print_result("Interpretation", result.get('interpretation'))
        print_result("Annualized Return %", result.get('avg_return_annualized'))
        print_result("Annualized Vol %", result.get('volatility_annualized'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "What's the max drawdown for Tesla?"
    print("\nQuery: What's the max drawdown for Tesla?")
    result = await av_max_drawdown_impl("TSLA")
    if print_test("av_max_drawdown(TSLA)", result):
        print_result("Max Drawdown %", result.get('max_drawdown_percent'))
        print_result("Current Drawdown %", result.get('current_drawdown_percent'))
        print_result("Assessment", result.get('status_assessment'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # TEST 10: FX Tools
    # =========================================================================
    print_header("10. FX-Specific Tools")

    # Natural language query: "What's the EUR/USD exchange rate?"
    print("\nQuery: What's the EUR/USD exchange rate?")
    result = await av_fx_quote_impl("EUR", "USD")
    if print_test("av_fx_quote(EUR/USD)", result):
        print_result("Exchange Rate", result.get('exchange_rate'))
        print_result("Bid", result.get('bid'))
        print_result("Ask", result.get('ask'))
        print_result("Spread", result.get('spread'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "Give me technical analysis for GBP/USD"
    print("\nQuery: Give me technical analysis for GBP/USD")
    result = await av_fx_technical_impl("GBP", "USD")
    if print_test("av_fx_technical(GBP/USD)", result):
        print_result("Current Rate", result.get('current_rate'))
        print_result("Overall Signal", result.get('overall_signal'))
        indicators = result.get('indicators', {})
        print_result("RSI", indicators.get('rsi'))
        print_result("ATR (pips)", indicators.get('atr_pips'))
        passed += 1
    else:
        failed += 1

    # Natural language query: "What are the pivot points for USD/JPY?"
    print("\nQuery: What are the pivot points for USD/JPY?")
    result = await av_fx_pivot_impl("USD", "JPY")
    if print_test("av_fx_pivot(USD/JPY)", result):
        print_result("Current Rate", result.get('current_rate'))
        pivots = result.get('pivot_points', {})
        print_result("Pivot", pivots.get('pivot'))
        print_result("R1", pivots.get('r1'))
        print_result("S1", pivots.get('s1'))
        print_result("Position", result.get('position'))
        passed += 1
    else:
        failed += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("Test Summary")

    total = passed + failed
    print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
    print(f"  {Colors.GREEN}Passed:{Colors.RESET} {passed}/{total}")
    print(f"  {Colors.RED}Failed:{Colors.RESET} {failed}/{total}")
    print(f"  {Colors.YELLOW}Success Rate:{Colors.RESET} {(passed/total*100):.1f}%")

    # Rate limit status
    rate_status = get_rate_limit_status()
    print(f"\n{Colors.BOLD}API Rate Limit:{Colors.RESET}")
    print(f"  Remaining: {rate_status['remaining']}/{rate_status['max_calls']} calls/minute")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return passed, failed


if __name__ == '__main__':
    passed, failed = asyncio.run(run_tests())
    sys.exit(0 if failed == 0 else 1)
