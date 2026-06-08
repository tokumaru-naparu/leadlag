output/results/ - 数値結果CSVファイル

paper_trade_log.csv
  書き込み: scripts/08_signal_today.py（毎日実行）
  読み込み: scripts/send_daily_email_report.py
  内容: 日次シグナル・ポジション・DD情報のログ

results.csv
  書き込み: バックテスト実行時
  読み込み: scripts/extract_positions_and_compute_sizes.py
  内容: バックテスト日次シグナル結果
