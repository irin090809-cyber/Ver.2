import numpy as np
import pandas as pd
from src.indicators.technical_indicators import detect_rsi_bullish_divergence


def _fit_trendline_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def detect_bottom_reversal_pattern(df: pd.DataFrame) -> dict:
    """장기 하락 → 바닥·횡보 → 추세선 돌파·거래량 급증 → 눌림목·지지 재반등 흐름 점검."""
    empty = {
        "downtrend_prior": False,
        "base_formed": False,
        "trendline_breakout": False,
        "volume_surge": False,
        "ichimoku_shift": False,
        "pullback_rebound": False,
        "bullish_divergence": False,
        "pattern_stage": "NONE",
    }
    if df is None or len(df) < 80:
        return empty

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    vol_ratio = df["vol_ratio"].to_numpy(dtype=float) if "vol_ratio" in df.columns else np.ones(len(df))

    # 1. 이전 구간 장기 하락 추세 확인 (최근 120봉 기준 앞쪽 구간)
    prior_slice = close[max(0, len(close) - 120) : max(20, len(close) - 45)]
    prior_peak = float(np.max(prior_slice)) if len(prior_slice) > 0 else float(close[-1])
    recent_low_min = float(np.min(low[-45:]))
    downtrend_prior = (prior_peak > 0) and ((prior_peak - recent_low_min) / prior_peak >= 0.20)

    # 2. 바닥 형성 및 횡보 확인 (최근 20~45봉 구간 변동폭 축소 및 저점 방어)
    base_slice = close[-35:-10] if len(close) >= 35 else close[:-10]
    base_range_ratio = (
        (float(np.max(base_slice)) - float(np.min(base_slice))) / float(np.mean(base_slice))
        if len(base_slice) >= 5 and float(np.mean(base_slice)) > 0
        else 1.0
    )
    base_formed = downtrend_prior and (base_range_ratio <= 0.18)

    # 3. 하락 추세선 돌파 확인 (최근 30봉 고점들의 기울기가 음수인 추세선을 최근 봉 종가가 상향 돌파)
    line_window = high[-35:-3] if len(high) >= 35 else high[:-3]
    slope = _fit_trendline_slope(line_window)
    x_idx = len(line_window) + 2
    x_ref = np.arange(len(line_window), dtype=float)
    _, intercept = np.polyfit(x_ref, line_window, 1)
    projected_line = intercept + slope * x_idx
    recent_max_close = float(np.max(close[-5:]))
    trendline_breakout = (slope < 0) and (recent_max_close > projected_line)

    # 4. 거래량 급증 확인 (최근 10봉 내 20일 평균 대비 1.8배 이상 거래량 발생)
    recent_max_vol_ratio = float(np.max(vol_ratio[-10:])) if len(vol_ratio) >= 10 else 1.0
    volume_surge = recent_max_vol_ratio >= 1.8

    # 5. 일목균형표 추세 전환 확인 (구름대 상향 돌파 또는 전환선/기준선 골든크로스/양운)
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    tenkan_above_kijun = bool(last_row.get("tenkan", 0) >= last_row.get("kijun", 0))
    tenkan_cross = bool(
        (prev_row.get("tenkan", 0) <= prev_row.get("kijun", 0))
        and (last_row.get("tenkan", 0) > last_row.get("kijun", 0))
    )
    above_cloud = bool(last_row.get("above_cloud", False))
    ichimoku_shift = above_cloud or (tenkan_above_kijun and (tenkan_cross or bool(last_row.get("cloud_bullish", False))))

    # 6~7. 눌림목 형성 및 지지·재반등 확인
    # 최근 돌파 고점 이후 고가가 소폭 조정(2~15%)된 뒤 최근 2봉 종가가 전일보다 반등하고 주요 지지선(20일선 또는 기준선) 위 유지
    recent_high_15 = float(np.max(high[-15:]))
    pullback_depth = (recent_high_15 - float(close[-1])) / recent_high_15 if recent_high_15 > 0 else 0.0
    has_pullback = 0.015 <= pullback_depth <= 0.18
    rebound_now = (float(close[-1]) > float(close[-2])) and (float(close[-1]) >= float(open_val := df["Open"].iloc[-1]) * 0.995)
    support_ok = float(close[-1]) >= float(last_row.get("kijun", 0)) * 0.97 or float(close[-1]) >= float(last_row.get("ma_20", 0)) * 0.97
    pullback_rebound = has_pullback and rebound_now and support_ok

    # RSI 상승 다이버전스
    div_info = detect_rsi_bullish_divergence(df)
    bullish_div = bool(div_info.get("bullish_divergence", False))

    if pullback_rebound and trendline_breakout:
        stage = "PULLBACK_REBOUND"
    elif trendline_breakout and volume_surge:
        stage = "BREAKOUT_SURGE"
    elif base_formed and (bullish_div or ichimoku_shift):
        stage = "BASE_BUILDING"
    elif downtrend_prior:
        stage = "DOWNTREND"
    else:
        stage = "NONE"

    return {
        "downtrend_prior": bool(downtrend_prior),
        "base_formed": bool(base_formed),
        "trendline_breakout": bool(trendline_breakout),
        "volume_surge": bool(volume_surge),
        "ichimoku_shift": bool(ichimoku_shift),
        "pullback_rebound": bool(pullback_rebound),
        "bullish_divergence": bullish_div,
        "pattern_stage": stage,
    }
