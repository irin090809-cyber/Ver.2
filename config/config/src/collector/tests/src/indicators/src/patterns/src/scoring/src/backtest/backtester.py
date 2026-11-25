import numpy as np
import pandas as pd
from src.indicators.technical_indicators import build_technical_frame
from src.patterns.bottom_reversal import detect_bottom_reversal_pattern
from src.scoring.signal_scorer import score_and_build_trade_plan


def simulate_single_trade(
    future_df: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    target_1: float,
    target_2: float,
    max_hold_bars: int = 20,
) -> dict:
    """진입 이후 최대 보유 봉(기본 20거래일=약 4주 스윙) 동안 손절·1차·2차 목표가 도달 시뮬레이션."""
    if future_df is None or future_df.empty or entry_price <= 0:
        return {"return_pct": 0.0, "exit_reason": "NO_DATA", "hold_bars": 0}

    window = future_df.iloc[:max_hold_bars]
    hit_t1 = False
    realized_return = 0.0

    for idx_pos, (_, row) in enumerate(window.iterrows(), start=1):
        low_val = float(row["Low"])
        high_val = float(row["High"])
        close_val = float(row["Close"])

        # 보수적 접근: 같은 봉에서 손절가와 목표가가 함께 닿으면 손절 우선 처리
        if low_val <= stop_loss:
            stop_ret = (stop_loss - entry_price) / entry_price
            if hit_t1:
                # 1차 절반 익절 후 잔량 손절(본절 근처 정리)
                final_ret = 0.5 * ((target_1 - entry_price) / entry_price) + 0.5 * stop_ret
                return {"return_pct": round(final_ret * 100, 2), "exit_reason": "T1_THEN_STOP", "hold_bars": idx_pos}
            return {"return_pct": round(stop_ret * 100, 2), "exit_reason": "STOP_LOSS", "hold_bars": idx_pos}

        if not hit_t1 and high_val >= target_1:
            hit_t1 = True

        if hit_t1 and high_val >= target_2:
            t1_ret = (target_1 - entry_price) / entry_price
            t2_ret = (target_2 - entry_price) / entry_price
            avg_ret = 0.5 * t1_ret + 0.5 * t2_ret
            return {"return_pct": round(avg_ret * 100, 2), "exit_reason": "TARGET_1_2", "hold_bars": idx_pos}

        if idx_pos == len(window):
            end_ret = (close_val - entry_price) / entry_price
            if hit_t1:
                t1_ret = (target_1 - entry_price) / entry_price
                realized_return = 0.5 * t1_ret + 0.5 * end_ret
                reason = "T1_TIME_EXIT"
            else:
                realized_return = end_ret
                reason = "TIME_EXIT"
            return {"return_pct": round(realized_return * 100, 2), "exit_reason": reason, "hold_bars": idx_pos}

    return {"return_pct": 0.0, "exit_reason": "NONE", "hold_bars": 0}


def run_symbol_backtest(
    code: str,
    name: str,
    market: str,
    daily_df: pd.DataFrame,
    liquidity_risk: str = "LOW",
    avg_tv_20d: float = 0.0,
    min_score: float = 65.0,
    max_hold_bars: int = 20,
    step_bars: int = 5,
) -> list[dict]:
    """단일 종목 과거 구간을 일정 간격으로 훑으며 신호 발생 및 스윙 수익률 검증."""
    if daily_df is None or len(daily_df) < 120:
        return []

    full_tech = build_technical_frame(daily_df)
    trades: list[dict] = []

    start_idx = 90
    end_idx = len(full_tech) - max_hold_bars - 1
    if end_idx <= start_idx:
        return []

    for i in range(start_idx, end_idx, step_bars):
        hist_slice = full_tech.iloc[: i + 1]
        future_slice = full_tech.iloc[i + 1 : i + 1 + max_hold_bars]

        pattern = detect_bottom_reversal_pattern(hist_slice)
        plan = score_and_build_trade_plan(
            code=code,
            name=name,
            market=market,
            df=hist_slice,
            pattern=pattern,
            liquidity_risk=liquidity_risk,
            avg_tv_20d=avg_tv_20d,
        )
        if plan is None or plan["score"] < min_score:
            continue

        sim = simulate_single_trade(
            future_df=future_slice,
            entry_price=plan["entry_price"],
            stop_loss=plan["stop_loss"],
            target_1=plan["target_1"],
            target_2=plan["target_2"],
            max_hold_bars=max_hold_bars,
        )
        trades.append(
            {
                "signal_date": str(hist_slice.index[-1].date()),
                "code": code,
                "name": name,
                "market": market,
                "score": plan["score"],
                "grade": plan["grade"],
                "stage": plan["pattern_stage"],
                **sim,
            }
        )

    return trades


def summarize_backtest(trades: list[dict]) -> dict:
    """백테스트 거래 내역을 바탕으로 승률·평균수익·최대손실 등 요약 지표 계산."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "max_loss_pct": 0.0,
            "max_win_pct": 0.0,
        }

    returns = np.array([float(t.get("return_pct", 0.0)) for t in trades], dtype=float)
    wins = returns > 0
    return {
        "total_trades": int(len(returns)),
        "win_rate_pct": round(float(np.mean(wins) * 100.0), 2),
        "avg_return_pct": round(float(np.mean(returns)), 2),
        "max_loss_pct": round(float(np.min(returns)), 2),
        "max_win_pct": round(float(np.max(returns)), 2),
    }
