# Daily Signal + Email Setup

This repository now includes a GitHub Actions workflow that:
1. Runs `scripts/08_signal_today.py` every weekday.
2. Builds a daily report from `output/results/paper_trade_log.csv`.
3. Sends the report by email.

## Added Files

- `.github/workflows/daily_signal_email.yml`
- `scripts/send_daily_email_report.py`

## Required GitHub Secrets

Set these in: `Settings > Secrets and variables > Actions > Secrets`

- `SMTP_HOST` (example: `smtp.gmail.com`)
- `SMTP_PORT` (example: `587`)
- `SMTP_USER`
- `SMTP_PASS`
- `MAIL_TO` (comma-separated allowed)

Optional:
- `MAIL_FROM` (if omitted, `SMTP_USER` is used)

## Fee Handling

`08_signal_today.py` now records fee estimates in the daily log:

- `fee_bps_per_side`
- `gross_notional`
- `fee_entry_est`
- `fee_roundtrip_est`

Default fee assumption is `5.0 bps per side`.
You can override by setting environment variable `TRADE_FEE_BPS_PER_SIDE` before running scripts.

## Schedule

Workflow schedule is set to weekdays 09:10 JST.
Manual run is also enabled via `workflow_dispatch`.

## Notes

- On weekends/holidays, `08_signal_today.py` may skip trading logic; the email still sends run status.
- The Actions runner does not persist local file updates unless you commit/push them. The email uses that run's generated output.
