from dataclasses import dataclass


@dataclass(frozen=True)
class CollectorConfig:
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ")
    lookback_days: int = 400
    weekly_rule: str = "W-FRI"
    min_bars_daily: int = 60
    min_bars_weekly: int = 12
    low_liquidity_avg_tv_20d: float = 1_000_000_000.0
    cache_dir: str = "data/cache"
    output_dir: str = "output"
    max_workers: int = 8
    request_retry: int = 2
    request_delay: float = 0.25
