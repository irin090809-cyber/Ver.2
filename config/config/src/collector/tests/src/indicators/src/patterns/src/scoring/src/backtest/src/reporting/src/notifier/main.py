from datetime import datetime
from config.settings import CollectorConfig
from src.backtest.backtester import run_symbol_backtest, summarize_backtest
from src.collector.data_collector import collect_universe_data
from src.indicators.technical_indicators import build_technical_frame
from src.notifier.telegram_notifier import dispatch_scan_results
from src.patterns.bottom_reversal import detect_bottom_reversal_pattern
from src.reporting.report_generator import generate_html_report
from src.scoring.signal_scorer import score_and_build_trade_plan, select_top_candidates


def run_scanner(
    config: CollectorConfig = CollectorConfig(),
    limit: int | None = None,
    run_backtest_sample: int = 10,
) -> list[dict]:
    date_tag = datetime.now().strftime("%Y%m%d")
    bundles = collect_universe_data(config=config, limit=limit)

    raw_candidates: list[dict] = []
    backtest_trades: list[dict] = []

    for idx, (code, bundle) in enumerate(bundles.items()):
        daily_df = bundle.get("daily")
        if daily_df is None or daily_df.empty:
            continue

        tech_df = build_technical_frame(daily_df)
        pattern = detect_bottom_reversal_pattern(tech_df)
        plan = score_and_build_trade_plan(
            code=code,
            name=bundle["name"],
            market=bundle["market"],
            df=tech_df,
            pattern=pattern,
            liquidity_risk=bundle.get("liquidity_risk", "LOW"),
            avg_tv_20d=bundle.get("avg_tv_20d", 0.0),
        )
        if plan is not None:
            raw_candidates.append(plan)

        if idx < run_backtest_sample:
            trades = run_symbol_backtest(
                code=code,
                name=bundle["name"],
                market=bundle["market"],
                daily_df=daily_df,
                liquidity_risk=bundle.get("liquidity_risk", "LOW"),
                avg_tv_20d=bundle.get("avg_tv_20d", 0.0),
            )
            backtest_trades.extend(trades)

    top_candidates = select_top_candidates(raw_candidates, top_n=5, min_score=50.0)
    _, chart_paths = generate_html_report(
        candidates=top_candidates,
        bundles=bundles,
        output_dir=config.output_dir,
        date_tag=date_tag,
    )
    dispatch_scan_results(
        candidates=top_candidates,
        chart_paths=chart_paths,
        date_tag=date_tag,
    )

    bt_summary = summarize_backtest(backtest_trades)
    print(f"[DONE] Candidates: {len(top_candidates)} | Backtest Summary: {bt_summary}")
    return top_candidates


if __name__ == "__main__":
    run_scanner()
