import pandas as pd
from config.settings import CollectorConfig
from src.collector.data_collector import (
    assess_liquidity_risk,
    build_weekly_ohlcv,
    clean_daily_ohlcv,
    collect_universe_data,
)


def test_clean_and_weekly_resampling():
    idx = pd.date_range("2026-01-05", periods=10, freq="B")
    raw = pd.DataFrame(
        {
            "Open": [100 + i for i in range(10)],
            "High": [105 + i for i in range(10)],
            "Low": [98 + i for i in range(10)],
            "Close": [102 + i for i in range(10)],
            "Volume": [10000 * (i + 1) for i in range(10)],
        },
        index=idx,
    )
    daily = clean_daily_ohlcv(raw)
    assert len(daily) == 10
    assert "trading_value" in daily.columns

    weekly = build_weekly_ohlcv(daily)
    assert len(weekly) >= 2
    assert weekly["Volume"].sum() == daily["Volume"].sum()

    liq = assess_liquidity_risk(daily, threshold_tv=1_000_000_000)
    assert liq["liquidity_risk"] in {"LOW", "MEDIUM", "HIGH"}


def test_sample_universe_collection():
    cfg = CollectorConfig(min_bars_daily=20, min_bars_weekly=4)
    data = collect_universe_data(config=cfg, limit=5)
    assert isinstance(data, dict)
