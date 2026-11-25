import numpy as np
import pandas as pd


def score_and_build_trade_plan(
    code: str,
    name: str,
    market: str,
    df: pd.DataFrame,
    pattern: dict,
    liquidity_risk: str = "LOW",
    avg_tv_20d: float = 0.0,
    min_rr_ratio: float = 2.0,
) -> dict | None:
    """패턴·지표 종합 점수(100점), 등급(A~D), 매매 계획(진입·손절·목표가·손익비) 산출."""
    if df is None or len(df) < 30 or not pattern:
        return None

    last = df.iloc[-1]
    entry = float(last["Close"])
    if entry <= 0:
        return None

    # 최근 15봉 주요 저점 기준 손절가 설정 (최소 2%, 최대 12% 이내로 제한)
    recent_swing_low = float(np.min(df["Low"].iloc[-15:].to_numpy(dtype=float)))
    raw_stop = recent_swing_low * 0.985
    stop_loss = min(raw_stop, entry * 0.98)
    stop_loss = max(stop_loss, entry * 0.88)

    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return None

    # 최근 60봉 고점 및 120봉 고점 참고하여 1차(1:2 이상), 2차(1:3 이상) 목표가 산정
    high_60 = float(np.max(df["High"].iloc[-60:].to_numpy(dtype=float)))
    high_120 = float(np.max(df["High"].iloc[-120:].to_numpy(dtype=float)))

    min_target_1 = entry + risk_per_share * min_rr_ratio
    target_1 = max(min_target_1, high_60 * 0.98 if high_60 > min_target_1 * 1.03 else min_target_1)

    min_target_2 = entry + risk_per_share * max(3.0, min_rr_ratio + 1.0)
    target_2 = max(min_target_2, high_120 * 0.98 if high_120 > min_target_2 * 1.05 else min_target_2)

    rr_ratio_1 = round((target_1 - entry) / risk_per_share, 2)
    rr_ratio_2 = round((target_2 - entry) / risk_per_share, 2)

    # 손익비 기준 미달 시 제외
    if rr_ratio_1 < min_rr_ratio:
        return None

    # 100점 만점 점수 계산
    score = 0.0
    if pattern.get("downtrend_prior"):
        score += 12.0
    if pattern.get("base_formed"):
        score += 13.0
    if pattern.get("trendline_breakout"):
        score += 20.0
    if pattern.get("volume_surge"):
        score += 15.0
    if pattern.get("ichimoku_shift"):
        score += 15.0
    if pattern.get("pullback_rebound"):
        score += 15.0
    if pattern.get("bullish_divergence"):
        score += 10.0

    # 유동성 리스크 반영 (강제 배제는 아니되 점수에 차등 반영)
    if liquidity_risk == "HIGH":
        score -= 10.0
    elif liquidity_risk == "MEDIUM":
        score -= 3.0

    score = float(np.clip(round(score, 1), 0.0, 100.0))

    if score >= 80:
        grade = "A"
    elif score >= 65:
        grade = "B"
    elif score >= 50:
        grade = "C"
    else:
        grade = "D"

    return {
        "code": code,
        "name": name,
        "market": market,
        "score": score,
        "grade": grade,
        "pattern_stage": pattern.get("pattern_stage", "NONE"),
        "entry_price": round(entry, 0),
        "stop_loss": round(stop_loss, 0),
        "target_1": round(target_1, 0),
        "target_2": round(target_2, 0),
        "rr_ratio_1": rr_ratio_1,
        "rr_ratio_2": rr_ratio_2,
        "liquidity_risk": liquidity_risk,
        "avg_tv_20d": round(float(avg_tv_20d), 0),
        "pullback_rebound": bool(pattern.get("pullback_rebound", False)),
        "bullish_divergence": bool(pattern.get("bullish_divergence", False)),
    }


def select_top_candidates(
    candidates: list[dict],
    top_n: int = 5,
    min_score: float = 50.0,
) -> list[dict]:
    """점수순 정렬 후 눌림목 재반등·유동성 양호 우선으로 상위 최대 5개 선별."""
    filtered = [c for c in candidates if c.get("score", 0.0) >= min_score]
    ranked = sorted(
        filtered,
        key=lambda item: (
            item.get("score", 0.0),
            1 if item.get("pullback_rebound") else 0,
            0 if item.get("liquidity_risk") == "HIGH" else 1,
            item.get("rr_ratio_1", 0.0),
        ),
        reverse=True,
    )
    return ranked[:top_n]
