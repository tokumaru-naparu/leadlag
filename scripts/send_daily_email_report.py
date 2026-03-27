"""
send_daily_email_report.py
GitHub Actionsで実行する日次メール送信スクリプト（3/20からの累積報告版）。

前提:
- scripts/08_signal_today.py 実行後に output/results/paper_trade_log.csv が更新される
- output/results/test_mar20_to_today.csv に 3/20からの日次パフォーマンスが保存される

必要な環境変数:
- SMTP_HOST
- SMTP_PORT (例: 587)
- SMTP_USER
- SMTP_PASS
- MAIL_TO (カンマ区切り可)

任意の環境変数:
- MAIL_FROM (未指定時はSMTP_USER)
- TRADE_FEE_BPS_PER_SIDE (未指定時 5.0)
"""

from __future__ import annotations

import datetime as dt
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "output" / "results"
LOG_CSV = RESULTS_DIR / "paper_trade_log.csv"
PERF_CSV = RESULTS_DIR / "test_mar20_to_today.csv"


def get_env(name: str, required: bool = True, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Environment variable missing: {name}")
    return value if value is not None else ""


def safe_float(v) -> float | None:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def load_cumulative_performance() -> dict:
    """3/20からの累積パフォーマンスを読み込む"""
    if not PERF_CSV.exists():
        return {"status": "perf_error", "message": "test_mar20_to_today.csv が見つかりません"}
    
    df = pd.read_csv(PERF_CSV)
    if len(df) == 0:
        return {"status": "perf_error", "message": "CSVが空です"}
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # 累積リターン計算
    start_capital = 1.0
    final_capital = None
    if "cum_curve" in df.columns:
        final_capital = df["cum_curve"].iloc[-1]
    
    cum_return_pct = (final_capital - start_capital) * 100 if final_capital else 0
    
    # 勝率計算
    wins = 0
    total = 0
    if "strategy_return" in df.columns:
        wins = (df["strategy_return"] > 0).sum()
        total = len(df)
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # 本日分（最新行）
    latest = df.iloc[-1].to_dict()
    today_return = safe_float(latest.get("strategy_return", 0))
    today_long = safe_float(latest.get("long_return", 0))
    today_short = safe_float(latest.get("short_return", 0))
    
    return {
        "status": "OK",
        "cum_return_pct": cum_return_pct,
        "win_rate": win_rate,
        "total_days": total,
        "today_return": today_return * 100 if today_return else 0,
        "today_long": today_long * 100 if today_long else 0,
        "today_short": today_short * 100 if today_short else 0,
        "wins": wins,
        "df": df
    }


def load_latest_position() -> dict[str, str]:
    """paper_trade_log.csvから現在のポジション情報を取得"""
    if not LOG_CSV.exists():
        return {"status": "paper_trade_log.csv が見つかりません"}

    df = pd.read_csv(LOG_CSV)
    if len(df) == 0:
        return {"status": "paper_trade_log.csv は空です"}

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    row = df.iloc[-1].to_dict()
    out: dict[str, str] = {}

    row_date = pd.to_datetime(row.get("date"), errors="coerce")
    out["status"] = "OK"
    out["row_date"] = row_date.date().isoformat() if pd.notna(row_date) else "N/A"

    for k in [
        "signal_strength",
        "signal_pct",
        "final_size",
        "dd_scale",
        "current_dd",
        "capital",
        "long_1",
        "long_2",
        "long_3",
        "short_1",
        "short_2",
        "short_3",
    ]:
        out[k] = "" if k not in row else str(row.get(k, ""))

    return out


def build_mail_body(perf: dict, position: dict, fee_bps_per_side: float) -> str:
    """3/20からの累積レポート + 本日のポジション情報"""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("LeadLag Daily Report")
    lines.append(f"3月20日 ~ 本日のパフォーマンス")
    lines.append(f"Generated at: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 60)
    lines.append("")
    
    if perf.get("status") != "OK":
        lines.append(f"⚠ Error: {perf.get('message', 'Unknown error')}")
        return "\n".join(lines)
    
    # 累積リターン
    cum_return = perf.get("cum_return_pct", 0)
    sign = "+" if cum_return >= 0 else ""
    lines.append(f"【累積パフォーマンス (3/20 ~ 本日)】")
    lines.append(f"累積リターン: {sign}{cum_return:.2f}%")
    lines.append(f"取引日数: {perf.get('total_days', 0)} 日")
    lines.append(f"勝率: {perf.get('win_rate', 0):.1f}% ({perf.get('wins', 0)} / {perf.get('total_days', 0)})")
    lines.append("")
    
    # 本日のパフォーマンス
    today_ret = perf.get("today_return", 0)
    today_long = perf.get("today_long", 0)
    today_short = perf.get("today_short", 0)
    
    sign_ret = "+" if today_ret >= 0 else ""
    sign_long = "+" if today_long >= 0 else ""
    sign_short = "+" if today_short >= 0 else ""
    
    lines.append(f"【本日のパフォーマンス】")
    lines.append(f"総リターン: {sign_ret}{today_ret:.3f}%")
    lines.append(f"  ロング側:   {sign_long}{today_long:.3f}%")
    lines.append(f"  ショート側: {sign_short}{today_short:.3f}%")
    lines.append("")
    
    # 現在のポジション
    if position.get("status") == "OK":
        lines.append(f"【現在のポジション】")
        lines.append(f"信号強度: {position.get('signal_strength', '')}")
        lines.append(f"信号パーセンタイル: {position.get('signal_pct', '')}%")
        lines.append(f"ロング（3セクター）: {position.get('long_1', '')}, {position.get('long_2', '')}, {position.get('long_3', '')}")
        lines.append(f"ショート（3セクター）: {position.get('short_1', '')}, {position.get('short_2', '')}, {position.get('short_3', '')}")
        lines.append(f"ポジションサイズ: {position.get('final_size', '')}x")
        lines.append(f"ドローダウン: {position.get('current_dd', '')}")
        lines.append(f"ドローダウン調整倍率: {position.get('dd_scale', '')}x")
    lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def send_mail(subject: str, body: str) -> None:
    smtp_host = get_env("SMTP_HOST")
    smtp_port = int(get_env("SMTP_PORT", default="587"))
    smtp_user = get_env("SMTP_USER")
    smtp_pass = get_env("SMTP_PASS")
    mail_to = get_env("MAIL_TO")
    mail_from = get_env("MAIL_FROM", required=False, default=smtp_user)

    recipients = [x.strip() for x in mail_to.split(",") if x.strip()]
    if not recipients:
        raise RuntimeError("MAIL_TO is empty")

    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


def main() -> None:
    fee_bps_per_side = float(os.getenv("TRADE_FEE_BPS_PER_SIDE", "5.0"))
    
    perf = load_cumulative_performance()
    position = load_latest_position()
    
    perf_date = dt.date.today().isoformat()
    if perf.get("status") == "OK" and len(perf.get("df", [])) > 0:
        latest_date = perf["df"].iloc[-1].get("date")
        if hasattr(latest_date, "date"):
            perf_date = latest_date.date().isoformat()
        elif isinstance(latest_date, str):
            perf_date = latest_date
    
    subject = f"[LeadLag] {perf_date} 日次報告"
    body = build_mail_body(perf=perf, position=position, fee_bps_per_side=fee_bps_per_side)
    
    send_mail(subject=subject, body=body)
    print("Mail sent successfully")


if __name__ == "__main__":
    main()
