import html
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_candidates_csv(candidates: list[dict], output_dir: str, date_tag: str) -> str:
    """선별된 상위 종목 결과를 CSV 파일로 저장."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"candidates_{date_tag}.csv")
    if not candidates:
        pd.DataFrame().to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

    df = pd.DataFrame(candidates)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def render_candidate_chart(
    candidate: dict,
    daily_df: pd.DataFrame,
    output_dir: str,
    date_tag: str,
    bars: int = 90,
) -> str | None:
    """종목별 최근 일봉 종가·이평선 및 진입/손절/목표가 차트 PNG 생성."""
    if daily_df is None or daily_df.empty:
        return None

    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    code = candidate["code"]
    chart_path = os.path.join(charts_dir, f"{date_tag}_{code}.png")

    sub = daily_df.iloc[-bars:].copy()
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    x_vals = range(len(sub))
    ax_price.plot(x_vals, sub["Close"], label="Close", color="#111827", linewidth=1.6)
    for w, color in ((20, "#2563eb"), (60, "#d97706"), (120, "#7c3aed")):
        col = f"ma_{w}"
        if col in sub.columns:
            ax_price.plot(x_vals, sub[col], label=f"MA{w}", color=color, linewidth=1.0, alpha=0.85)

    entry = float(candidate["entry_price"])
    stop = float(candidate["stop_loss"])
    t1 = float(candidate["target_1"])
    t2 = float(candidate["target_2"])

    ax_price.axhline(entry, color="#059669", linestyle="-", linewidth=1.0, label=f"Entry {entry:,.0f}")
    ax_price.axhline(stop, color="#dc2626", linestyle="--", linewidth=1.0, label=f"Stop {stop:,.0f}")
    ax_price.axhline(t1, color="#0284c7", linestyle=":", linewidth=1.1, label=f"Target1 {t1:,.0f}")
    ax_price.axhline(t2, color="#4f46e5", linestyle=":", linewidth=1.1, label=f"Target2 {t2:,.0f}")

    ax_price.set_title(
        f"[{code}] {candidate['market']} | Grade {candidate['grade']} ({candidate['score']}pt) | {candidate['pattern_stage']}",
        fontsize=11,
        fontweight="bold",
    )
    ax_price.legend(loc="upper left", fontsize=8, ncol=3)
    ax_price.grid(True, linestyle="--", alpha=0.3)

    ax_vol.bar(x_vals, sub["Volume"], color="#64748b", alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=9)
    ax_vol.grid(True, linestyle="--", alpha=0.3)

    step = max(1, len(sub) // 6)
    tick_idx = list(range(0, len(sub), step))
    tick_labels = [str(sub.index[i].date()) for i in tick_idx]
    ax_vol.set_xticks(tick_idx)
    ax_vol.set_xticklabels(tick_labels, rotation=20, fontsize=8)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=130)
    plt.close(fig)
    return chart_path


def generate_html_report(
    candidates: list[dict],
    bundles: dict[str, dict],
    output_dir: str,
    date_tag: str,
) -> tuple[str, list[str]]:
    """CSV, 종목별 차트 이미지, 종합 HTML 보고서를 생성하고 (html_path, chart_paths) 반환."""
    os.makedirs(output_dir, exist_ok=True)
    save_candidates_csv(candidates, output_dir, date_tag)

    chart_paths: list[str] = []
    cards_html: list[str] = []

    for item in candidates:
        code = item["code"]
        daily_df = bundles.get(code, {}).get("daily")
        c_path = render_candidate_chart(item, daily_df, output_dir, date_tag)
        rel_img = ""
        if c_path:
            chart_paths.append(c_path)
            rel_img = f'<img src="charts/{os.path.basename(c_path)}" style="max-width:100%;border-radius:8px;margin-top:10px;" />'

        cards_html.append(
            f"""
            <div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:18px;background:#ffffff;">
              <h3 style="margin:0 0 8px 0;">{html.escape(item['name'])} ({html.escape(code)}) · {html.escape(item['market'])}</h3>
              <p style="margin:4px 0;"><strong>등급 / 점수:</strong> {html.escape(item['grade'])}등급 ({item['score']}점) · <strong>단계:</strong> {html.escape(item['pattern_stage'])} · <strong>유동성 리스크:</strong> {html.escape(item['liquidity_risk'])}</p>
              <p style="margin:4px 0;"><strong>진입가:</strong> {item['entry_price']:,.0f}원 | <strong>손절가:</strong> {item['stop_loss']:,.0f}원 | <strong>1차 목표:</strong> {item['target_1']:,.0f}원 (1:{item['rr_ratio_1']}) | <strong>2차 목표:</strong> {item['target_2']:,.0f}원 (1:{item['rr_ratio_2']})</p>
              {rel_img}
            </div>
            """
        )

    body_content = (
        "".join(cards_html)
        if cards_html
        else "<p>금일 조건을 충족한 바닥권 추세 전환 후보 종목이 없습니다.</p>"
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>바닥권 추세 전환 스윙 스캐너 보고서 ({date_tag})</title>
</head>
<body style="font-family:sans-serif;max-width:920px;margin:24px auto;padding:0 16px;background:#f8fafc;color:#0f172a;">
  <h2>바닥권 추세 전환 스윙 스캐너 리포트 ({date_tag})</h2>
  <p>KOSPI + KOSDAQ 전 종목 대상 · 1~4주 스윙 관점 상위 후보 (최대 5종목)</p>
  {body_content}
</body>
</html>
"""
    html_path = os.path.join(output_dir, f"report_{date_tag}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    return html_path, chart_paths
