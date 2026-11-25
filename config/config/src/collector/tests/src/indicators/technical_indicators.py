import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in (5, 20, 60, 120):
        out[f"ma_{w}"] = out["Close"].rolling(w, min_periods=1).mean()
    out["vol_ma_20"] = out["Volume"].rolling(20, min_periods=1).mean()
    out["vol_ratio"] = np.where(
        out["vol_ma_20"] > 0, out["Volume"] / out["vol_ma_20"], 0.0
    )
    return out


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """일목균형표(전환선 9, 기준선 26, 선행스팬1/2 52, 후행스팬 26) 계산."""
    out = df.copy()
    high_9 = out["High"].rolling(9, min_periods=1).max()
    low_9 = out["Low"].rolling(9, min_periods=1).min()
    out["tenkan"] = (high_9 + low_9) / 2.0

    high_26 = out["High"].rolling(26, min_periods=1).max()
    low_26 = out["Low"].rolling(26, min_periods=1).min()
    out["kijun"] = (high_26 + low_26) / 2.0

    out["senkou_a"] = ((out["tenkan"] + out["kijun"]) / 2.0).shift(26)

    high_52 = out["High"].rolling(52, min_periods=1).max()
    low_52 = out["Low"].rolling(52, min_periods=1).min()
    out["senkou_b"] = ((high_52 + low_52) / 2.0).shift(26)

    out["chikou"] = out["Close"].shift(-26)

    cloud_top = out[["senkou_a", "senkou_b"]].max(axis=1)
    cloud_bot = out[["senkou_a", "senkou_b"]].min(axis=1)
    out["above_cloud"] = out["Close"] > cloud_top
    out["in_cloud"] = (out["Close"] >= cloud_bot) & (out["Close"] <= cloud_top)
    out["cloud_bullish"] = out["senkou_a"] >= out["senkou_b"]
    return out


def add_rsi_and_macd(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def detect_rsi_bullish_divergence(
    df: pd.DataFrame, lookback: int = 35, min_gap: int = 5
) -> dict:
    """최근 구간에서 주가 저점은 낮아지거나 비슷하고, RSI 저점은 높아지는 상승 다이버전스 탐지."""
    default_res = {"bullish_divergence": False, "prev_low": None, "last_low": None}
    if df is None or len(df) < lookback or "rsi" not in df.columns:
        return default_res

    sub = df.iloc[-lookback:].copy()
    lows = sub["Low"].to_numpy()
    rsis = sub["rsi"].to_numpy()

    pivot_idx = []
    for i in range(1, len(sub) - 1):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1] and not np.isnan(rsis[i]):
            pivot_idx.append(i)

    if len(pivot_idx) < 2:
        return default_res

    idx1, idx2 = pivot_idx[-2], pivot_idx[-1]
    if (idx2 - idx1) < min_gap:
        return default_res

    price_lower_or_equal = lows[idx2] <= lows[idx1] * 1.015
    rsi_higher = rsis[idx2] > rsis[idx1] + 1.5
    rsi_oversold_zone = min(rsis[idx1], rsis[idx2]) <= 45.0

    if price_lower_or_equal and rsi_higher and rsi_oversold_zone:
        return {
            "bullish_divergence": True,
            "prev_low": float(lows[idx1]),
            "last_low": float(lows[idx2]),
        }
    return default_res


def build_technical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_moving_averages(df)
    out = add_ichimoku(out)
    out = add_rsi_and_macd(out)
    return out
