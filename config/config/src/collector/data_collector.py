from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
import FinanceDataReader as fdr
import pandas as pd
from config.settings import CollectorConfig


def _retry_call(func, *args, retries: int = 2, delay: float = 0.25, **kwargs):
    last_err = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_err


def fetch_krx_universe(markets: tuple[str, ...] = ("KOSPI", "KOSDAQ")) -> pd.DataFrame:
    df = _retry_call(fdr.StockListing, "KRX", retries=2, delay=0.4)
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "market"])

    df = df[df["Market"].isin(markets)].copy()
    exclude_keywords = "스팩|ETF|ETN|리츠"
    df = df[~df["Name"].str.contains(exclude_keywords, na=False)]
    df = df[~df["Name"].str.endswith(("우", "우B", "우C"), na=False)]
    df = df[~df["Name"].str.contains(r"우\(", regex=True, na=False)]

    out = df[["Code", "Name", "Market"]].rename(
        columns={"Code": "code", "Name": "name", "Market": "market"}
    )
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.drop_duplicates(subset=["code"]).reset_index(drop=True)


def clean_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    clean = df.copy()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in clean.columns:
            return pd.DataFrame()
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["Open", "High", "Low", "Close"])
    clean["Volume"] = clean["Volume"].fillna(0)
    clean = clean[(clean["Close"] > 0) & (clean["High"] >= clean["Low"])]

    clean["trading_value"] = clean["Close"] * clean["Volume"]
    clean["avg_tv_20d"] = clean["trading_value"].rolling(20, min_periods=1).mean()
    clean["avg_vol_20d"] = clean["Volume"].rolling(20, min_periods=1).mean()
    return clean


def build_weekly_ohlcv(daily_df: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    weekly = daily_df.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "trading_value": "sum",
        }
    )
    weekly = weekly.dropna(subset=["Open", "High", "Low", "Close"])
    weekly = weekly[weekly["Close"] > 0]
    if not weekly.empty:
        weekly["avg_tv_12w"] = weekly["trading_value"].rolling(12, min_periods=1).mean()
    return weekly


def assess_liquidity_risk(daily_df: pd.DataFrame, threshold_tv: float) -> dict:
    if daily_df is None or daily_df.empty:
        return {"liquidity_risk": "HIGH", "avg_tv_20d": 0.0, "zero_vol_ratio_20d": 1.0}

    recent_20 = daily_df.iloc[-20:]
    avg_tv = float(recent_20["trading_value"].mean())
    zero_vol_ratio = float((recent_20["Volume"] <= 0).mean())

    if avg_tv < threshold_tv or zero_vol_ratio >= 0.2:
        risk = "HIGH"
    elif avg_tv < threshold_tv * 3 or zero_vol_ratio > 0.05:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "liquidity_risk": risk,
        "avg_tv_20d": round(avg_tv, 0),
        "zero_vol_ratio_20d": round(zero_vol_ratio, 3),
    }


def collect_symbol_bundle(
    code: str,
    name: str,
    market: str,
    start_str: str,
    end_str: str,
    config: CollectorConfig,
) -> dict | None:
    try:
        raw_df = _retry_call(
            fdr.DataReader,
            code,
            start_str,
            end_str,
            retries=config.request_retry,
            delay=config.request_delay,
        )
        daily_df = clean_daily_ohlcv(raw_df)
        if len(daily_df) < config.min_bars_daily:
            return None

        weekly_df = build_weekly_ohlcv(daily_df, rule=config.weekly_rule)
        if len(weekly_df) < config.min_bars_weekly:
            return None

        liq = assess_liquidity_risk(daily_df, config.low_liquidity_avg_tv_20d)
        return {
            "code": code,
            "name": name,
            "market": market,
            "daily": daily_df,
            "weekly": weekly_df,
            **liq,
        }
    except Exception:
        return None


def collect_universe_data(
    config: CollectorConfig = CollectorConfig(),
    as_of_date: datetime | None = None,
    limit: int | None = None,
) -> dict[str, dict]:
    ref_date = as_of_date or datetime.today()
    start_str = (ref_date - timedelta(days=config.lookback_days)).strftime("%Y-%m-%d")
    end_str = ref_date.strftime("%Y-%m-%d")

    universe = fetch_krx_universe(config.markets)
    if universe.empty:
        return {}

    rows = universe.to_dict("records")
    if limit is not None:
        rows = rows[:limit]

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(
                collect_symbol_bundle,
                r["code"],
                r["name"],
                r["market"],
                start_str,
                end_str,
                config,
            ): r["code"]
            for r in rows
        }
        for fut in as_completed(futures):
            bundle = fut.result()
            if bundle is not None:
                results[bundle["code"]] = bundle

    return results
