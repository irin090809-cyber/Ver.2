import os
import requests


def send_telegram_message(text: str, token: str | None = None, chat_id: str | None = None) -> bool:
    """텔레그램 텍스트 메시지 전송 (4096자 초과 시 자동 분할)."""
    bot_token = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    target_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not target_chat or not text:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    max_len = 3800
    chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]

    all_ok = True
    for chunk in chunks:
        try:
            resp = requests.post(
                url,
                json={"chat_id": target_chat, "text": chunk},
                timeout=15,
            )
            if not resp.ok:
                all_ok = False
        except Exception:
            all_ok = False
    return all_ok


def send_telegram_photo(
    photo_path: str,
    caption: str = "",
    token: str | None = None,
    chat_id: str | None = None,
) -> bool:
    """텔레그램 차트 이미지(PNG) 전송."""
    bot_token = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    target_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not target_chat or not photo_path or not os.path.exists(photo_path):
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    try:
        with open(photo_path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": target_chat, "caption": caption[:1000]},
                files={"photo": f},
                timeout=20,
            )
        return bool(resp.ok)
    except Exception:
        return False


def build_briefing_message(candidates: list[dict], date_tag: str) -> str:
    """선별된 상위 종목(최대 5개) 텔레그램 브리핑 텍스트 구성."""
    header = f"[바닥권 추세 전환 스윙 스캐너] ({date_tag})\nKOSPI+KOSDAQ · 1~4주 스윙 후보\n"
    if not candidates:
        return header + "\n금일 기준 조건을 충족한 후보 종목이 없습니다."

    lines = [header]
    for idx, item in enumerate(candidates[:5], start=1):
        lines.append(
            f"{idx}. {item['name']} ({item['code']}) [{item['market']}] - {item['grade']}등급 ({item['score']}점)\n"
            f"   단계: {item['pattern_stage']} | 유동성: {item['liquidity_risk']}\n"
            f"   진입: {item['entry_price']:,.0f}원 | 손절: {item['stop_loss']:,.0f}원\n"
            f"   1차목표: {item['target_1']:,.0f}원 (1:{item['rr_ratio_1']}) | 2차목표: {item['target_2']:,.0f}원 (1:{item['rr_ratio_2']})"
        )
    return "\n".join(lines)


def dispatch_scan_results(
    candidates: list[dict],
    chart_paths: list[str],
    date_tag: str,
) -> bool:
    """브리핑 메시지 및 차트 이미지를 순차 발송."""
    msg = build_briefing_message(candidates, date_tag)
    sent_text = send_telegram_message(msg)

    for c_path in chart_paths[:5]:
        base_name = os.path.basename(c_path)
        send_telegram_photo(c_path, caption=f"{date_tag} 차트 - {base_name}")

    return sent_text
