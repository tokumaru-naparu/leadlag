"""
send_daily_email_report.py
GitHub Actionsで実行する日次メール送信スクリプト。

前提:
- scripts/08_signal_today.py 実行後に output/results/paper_trade_log.csv が更新される
- SMTP情報は環境変数で渡す

必要な環境変数:
- SMTP_HOST
- SMTP_PORT (例: 587)
- SMTP_USER
- SMTP_PASS
- MAIL_TO (カンマ区切り可)

任意の環境変数:
- MAIL_FROM (未指定時はSMTP_USER)
- TRADE_FEE_BPS_PER_SIDE (未指定時 5.0)
- DAILY_RUN_LOG_PATH (未指定時 output/results/daily_run_output.txt)
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


def get_env(name: str, required: bool = True, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Environment variable missing: {name}")
    return value if value is not None else ""


def format_float(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def safe_float(v) -> float | None:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def calc_fee_estimates(capital: float, final_size: float, fee_bps_per_side: float) -> tuple[float, float, float]:
    fee_rate_side = fee_bps_per_side / 10000.0
    gross_notional = capital * final_size * 2.0
    entry_fee = gross_notional * fee_rate_side
    roundtrip_fee = gross_notional * fee_rate_side * 2.0
    return gross_notional, entry_fee, roundtrip_fee


def load_latest_row() -> dict[str, str]:
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

    today = dt.date.today()
    row_date = pd.to_datetime(row.get("date"), errors="coerce")
    out["status"] = "OK"
    out["row_date"] = row_date.date().isoformat() if pd.notna(row_date) else "N/A"
    out["today"] = today.isoformat()
    out["is_today"] = str(pd.notna(row_date) and row_date.date() == today)

    for k in [
        "signal_strength",
        "signal_pct",
        "final_size",
        "dd_scale",
        "current_dd",
        "capital",
        "peak",
        "long_1",
        "long_2",
        "long_3",
        "short_1",
        "short_2",
        "short_3",
        "actual_return",
        "fee_bps_per_side",
        "gross_notional",
        "fee_entry_est",
        "fee_roundtrip_est",
    ]:
        out[k] = "" if k not in row else str(row.get(k, ""))

    return out


def build_mail_body(latest: dict[str, str], run_log_text: str, fee_bps_per_side: float) -> str:
    lines: list[str] = []
    lines.append("LeadLag Daily Report")
    lines.append(f"Generated at: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    if latest.get("status") != "OK":
        lines.append(f"Status: {latest.get('status')}")
        lines.append("")
        lines.append("Run log:")
        lines.append(run_log_text.strip() if run_log_text.strip() else "(empty)")
        return "\n".join(lines)

    row_date = latest.get("row_date", "N/A")
    is_today = latest.get("is_today", "False")
    lines.append(f"Signal date in log: {row_date}")
    lines.append(f"Matches today: {is_today}")
    lines.append("")

    lines.append("Signal summary")
    lines.append(f"- signal_strength: {latest.get('signal_strength', '')}")
    lines.append(f"- percentile: {latest.get('signal_pct', '')}%")
    lines.append(
        f"- long: {latest.get('long_1', '')}, {latest.get('long_2', '')}, {latest.get('long_3', '')}"
    )
    lines.append(
        f"- short: {latest.get('short_1', '')}, {latest.get('short_2', '')}, {latest.get('short_3', '')}"
    )
    lines.append("")

    final_size = safe_float(latest.get("final_size"))
    capital = safe_float(latest.get("capital"))
    dd = safe_float(latest.get("current_dd"))
    dd_scale = safe_float(latest.get("dd_scale"))

    lines.append("Risk / sizing")
    if final_size is not None:
        lines.append(f"- final_size: {format_float(final_size, 3)}x")
    if dd is not None:
        lines.append(f"- current_dd: {dd * 100:.2f}%")
    if dd_scale is not None:
        lines.append(f"- dd_scale: {format_float(dd_scale, 2)}x")
    if capital is not None:
        lines.append(f"- capital: {capital:,.0f} JPY")
    lines.append("")

    lines.append("Fee estimates")
    fee_bps_logged = safe_float(latest.get("fee_bps_per_side"))
    fee_bps = fee_bps_logged if fee_bps_logged is not None else fee_bps_per_side

    gross_notional = safe_float(latest.get("gross_notional"))
    fee_entry_est = safe_float(latest.get("fee_entry_est"))
    fee_roundtrip_est = safe_float(latest.get("fee_roundtrip_est"))

    if (
        gross_notional is None
        or fee_entry_est is None
        or fee_roundtrip_est is None
    ) and capital is not None and final_size is not None:
        gross_notional, fee_entry_est, fee_roundtrip_est = calc_fee_estimates(
            capital=capital,
            final_size=final_size,
            fee_bps_per_side=fee_bps,
        )

    lines.append(f"- fee_bps_per_side: {fee_bps:.2f} bps")
    if gross_notional is not None:
        lines.append(f"- gross_notional(long+short): {gross_notional:,.0f} JPY")
    if fee_entry_est is not None:
        lines.append(f"- entry_fee_estimate: {fee_entry_est:,.0f} JPY")
    if fee_roundtrip_est is not None:
        lines.append(f"- roundtrip_fee_estimate: {fee_roundtrip_est:,.0f} JPY")
    lines.append("")

    actual_return = safe_float(latest.get("actual_return"))
    if actual_return is not None and final_size is not None and fee_bps is not None:
        # strategy returnからの簡易控除モデル: roundtrip fee rate = final_size*4*fee_rate_side
        fee_rate_side = fee_bps / 10000.0
        fee_rate_roundtrip = final_size * 4.0 * fee_rate_side
        net_actual = actual_return - fee_rate_roundtrip
        lines.append("Actual return check")
        lines.append(f"- actual_return(gross): {actual_return * 100:.3f}%")
        lines.append(f"- estimated_roundtrip_fee_rate: {fee_rate_roundtrip * 100:.3f}%")
        lines.append(f"- actual_return(net_est): {net_actual * 100:.3f}%")
        lines.append("")

    lines.append("Raw run log")
    lines.append(run_log_text.strip() if run_log_text.strip() else "(empty)")

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
    run_log_path = Path(
        os.getenv("DAILY_RUN_LOG_PATH", str(RESULTS_DIR / "daily_run_output.txt"))
    )
    run_log_text = ""
    if run_log_path.exists():
        run_log_text = run_log_path.read_text(encoding="utf-8", errors="ignore")

    latest = load_latest_row()
    subject_date = latest.get("row_date") or dt.date.today().isoformat()
    subject = f"[LeadLag] Daily Signal Report {subject_date}"
    body = build_mail_body(latest=latest, run_log_text=run_log_text, fee_bps_per_side=fee_bps_per_side)

    send_mail(subject=subject, body=body)
    print("Mail sent")


if __name__ == "__main__":
    main()
