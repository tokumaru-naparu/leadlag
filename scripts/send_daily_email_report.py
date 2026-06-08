"""
send_daily_email_report.py
GitHub Actionsで実行する日次メール送信スクリプト。

データソース:
- data/processed/performance.csv: 戦略パフォーマンス履歴（正統なデータソース）
- output/results/paper_trade_log.csv: 当日のポジション情報

必要な環境変数:
- SMTP_HOST
- SMTP_PORT (例: 587)
- SMTP_USER
- SMTP_PASS
- MAIL_TO (カンマ区切り可)

任意の環境変数:
- MAIL_FROM (未指定時はSMTP_USER)
- REPORT_START_DATE (集計開始日, 例: 2026-03-01。未指定時は当月1日)
"""

from __future__ import annotations

import datetime as dt
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PERF_CSV = ROOT / "data" / "processed" / "performance.csv"
LOG_CSV = ROOT / "output" / "results" / "paper_trade_log.csv"


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


def load_performance(start_date: dt.date) -> dict:
    """data/processed/performance.csv から指定日以降の戦略パフォーマンスを集計する"""
    if not PERF_CSV.exists():
        return {"status": "error", "message": f"{PERF_CSV} が見つかりません"}

    df = pd.read_csv(PERF_CSV, parse_dates=["date"])
    if len(df) == 0:
        return {"status": "error", "message": "performance.csv が空です"}

    df = df.sort_values("date")

    # 集計開始日の前日資本を基準にする
    start_ts = pd.Timestamp(start_date)
    df_before = df[df["date"] < start_ts]
    base_capital = float(df_before.iloc[-1]["capital"]) if len(df_before) > 0 else 1_000_000.0

    df_period = df[df["date"] >= start_ts].copy()
    if len(df_period) == 0:
        return {"status": "error", "message": f"{start_date} 以降のデータがありません"}

    final_capital = float(df_period.iloc[-1]["capital"])
    cum_return_pct = (final_capital - base_capital) / base_capital * 100

    wins = int((df_period["strategy_return"] > 0).sum())
    total = len(df_period)
    win_rate = wins / total * 100 if total > 0 else 0.0

    # 当月の最大日益・最大日損
    best_day_ret = float(df_period["strategy_return"].max()) * 100
    worst_day_ret = float(df_period["strategy_return"].min()) * 100
    best_day_date = df_period.loc[df_period["strategy_return"].idxmax(), "date"]
    worst_day_date = df_period.loc[df_period["strategy_return"].idxmin(), "date"]

    # 通算勝率（全期間）
    all_wins = int((df["strategy_return"] > 0).sum())
    all_total = len(df)
    all_win_rate = all_wins / all_total * 100 if all_total > 0 else 0.0

    # ドローダウン計算（全期間の資本推移から）
    capital_series = df["capital"]
    peak = capital_series.cummax()
    dd_series = (capital_series - peak) / peak * 100
    current_dd = float(dd_series.iloc[-1])
    max_dd_period = float(dd_series.min())

    latest = df_period.iloc[-1]
    today_return = safe_float(latest.get("strategy_return")) or 0.0
    today_long = safe_float(latest.get("long_return")) or 0.0
    today_short = safe_float(latest.get("short_return")) or 0.0
    latest_date = latest["date"]

    return {
        "status": "OK",
        "start_date": start_date.isoformat(),
        "latest_date": latest_date.date().isoformat() if hasattr(latest_date, "date") else str(latest_date),
        "cum_return_pct": cum_return_pct,
        "win_rate": win_rate,
        "total_days": total,
        "wins": wins,
        "today_return": today_return * 100,
        "today_long": today_long * 100,
        "today_short": today_short * 100,
        "base_capital": base_capital,
        "final_capital": final_capital,
        "best_day_ret": best_day_ret,
        "worst_day_ret": worst_day_ret,
        "best_day_date": best_day_date.date().isoformat() if hasattr(best_day_date, "date") else str(best_day_date),
        "worst_day_date": worst_day_date.date().isoformat() if hasattr(worst_day_date, "date") else str(worst_day_date),
        "all_win_rate": all_win_rate,
        "all_wins": all_wins,
        "all_total": all_total,
        "current_dd": current_dd,
        "max_dd_period": max_dd_period,
    }


def load_log() -> dict:
    """
    paper_trade_log.csvから以下を取得する:
    - 最新行: 今日計算したシグナル = 明日のポジション指示
    - 1つ前の行: 昨日のシグナルに基づく actual_return（手動入力済みの場合）
    """
    if not LOG_CSV.exists():
        return {"status": "no_log"}

    df = pd.read_csv(LOG_CSV)
    if len(df) == 0:
        return {"status": "empty"}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    def row_to_dict(row) -> dict:
        out = {}
        row_date = pd.to_datetime(row.get("date"), errors="coerce")
        out["row_date"] = row_date.date().isoformat() if pd.notna(row_date) else "N/A"
        for k in [
            "signal_strength", "signal_pct", "final_size", "dd_scale", "current_dd",
            "capital", "long_1", "long_2", "long_3", "short_1", "short_2", "short_3",
            "actual_return", "note",
        ]:
            out[k] = "" if k not in row else ("" if pd.isna(row.get(k, "")) else str(row.get(k, "")))
        return out

    result = {"status": "OK"}
    result["tomorrow"] = row_to_dict(df.iloc[-1])           # 明日のポジション（今日計算分）
    result["yesterday"] = row_to_dict(df.iloc[-2]) if len(df) >= 2 else None  # 昨日の実績
    return result


def _s(v: float) -> str:
    return f"+{v:.3f}%" if v >= 0 else f"{v:.3f}%"


SECTOR_JP = {
    "food":         "食品",
    "energy":       "エネルギー",
    "construction": "建設",
    "materials":    "素材化学",
    "pharma":       "医薬品",
    "auto":         "自動車",
    "steel":        "鉄鋼非鉄",
    "machinery":    "機械",
    "electronics":  "電機精密",
    "it_services":  "情報通信",
    "utilities":    "電力ガス",
    "transport":    "運輸物流",
    "trading":      "商社卸売",
    "retail":       "小売",
    "banks":        "銀行",
    "finance":      "金融",
    "realestate":   "不動産",
}


def build_mail_body(perf: dict, log: dict) -> str:
    if perf.get("status") != "OK":
        return f"ERROR: {perf.get('message', 'Unknown error')}"

    tr = perf["today_return"]
    tl = perf["today_long"]
    ts = perf["today_short"]
    win_mark = "○" if tr >= 0 else "×"

    # 月間累積リターン
    cum = perf.get("cum_return_pct", 0.0)
    start = perf.get("start_date", "")

    lines: list[str] = []
    lines.append(f"{perf['latest_date']}  {win_mark} {_s(tr)}")
    lines.append(f"  ロング:   {_s(tl)}")
    lines.append(f"  ショート: {_s(ts)}")
    lines.append(f"  {start}〜累積: {_s(cum)}")
    lines.append("")

    if log.get("status") == "OK" and log.get("tomorrow"):
        t = log["tomorrow"]
        size_str = t.get("final_size", "")
        try:
            size = float(size_str)
        except (ValueError, TypeError):
            size = None

        # TRADE_CAPITAL 環境変数（円）があれば金額も表示
        capital_str = os.getenv("TRADE_CAPITAL", "")
        try:
            trade_capital = float(capital_str)
        except (ValueError, TypeError):
            trade_capital = None

        def sname(key: str) -> str:
            return SECTOR_JP.get(key, key)

        long_names  = [sname(t.get(f"long_{i}",  "")) for i in (1, 2, 3)]
        short_names = [sname(t.get(f"short_{i}", "")) for i in (1, 2, 3)]

        lines.append(f"今日のポジション ({t.get('row_date','?')}寄り)")
        lines.append(f"  サイズ: {size_str}x")

        if size is not None and trade_capital is not None:
            notional = trade_capital * size
            each = notional / 3
            lines.append(f"  資金: {trade_capital:,.0f}円  建玉合計: {notional:,.0f}円  各業種: {each:,.0f}円")

        lines.append(f"  買い: {' / '.join(long_names)}")
        lines.append(f"  売り: {' / '.join(short_names)}")

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
    # 集計開始日: 環境変数で指定、なければ当月1日
    start_date_str = os.getenv("REPORT_START_DATE", "")
    if start_date_str:
        start_date = dt.date.fromisoformat(start_date_str)
    else:
        today = dt.date.today()
        start_date = today.replace(day=1)

    perf = load_performance(start_date)
    log = load_log()

    report_date = perf.get("latest_date", dt.date.today().isoformat())
    tr = perf.get("today_return", 0)
    win_mark = "○" if tr >= 0 else "×"
    subject = f"[LeadLag] {report_date} {win_mark} {'+' if tr >= 0 else ''}{tr:.2f}%"
    body = build_mail_body(perf=perf, log=log)

    send_mail(subject=subject, body=body)
    print("Mail sent successfully")


if __name__ == "__main__":
    main()
