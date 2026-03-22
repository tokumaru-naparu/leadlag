"""
run_all_analyses.py
1年・3年・5年・10年の全スパン分析を一括実行して
result_analysis/ にテキストファイルで保存する

実行すると:
  result_analysis/1year_analysis.txt
  result_analysis/3year_analysis.txt
  result_analysis/5year_analysis.txt
  result_analysis/10year_analysis.txt
  が作成される（10年分は既存データを使う）
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# ============================================================
# 設定
# ============================================================

BASE_DIR    = Path(__file__).parent
RESULT_DIR  = BASE_DIR / "result_analysis"
RESULT_DIR.mkdir(exist_ok=True)

PYTHON = sys.executable  # 現在のvenvのpython

# 分析するスパン（年数・月数・ラベル）
SPANS = [
    (1,   12,  "1year"),
    (3,   36,  "3year"),
    (5,   60,  "5year"),
    (10,  120, "10year"),
]

# 実行する分析スクリプト
ANALYSIS_SCRIPTS = [
    ("11_analyze_patterns.py",        "パターン分析（11）"),
    ("12_analyze_sectors.py",         "業種別統計（A）"),
    ("13_analyze_market_env.py",      "市場環境統計（B）"),
    ("14_analyze_pairs.py",           "業種ペア統計（C）"),
    ("15_analyze_signal_asymmetry.py","シグナル非対称性（D）"),
]

# ============================================================
# generate_history.py の MONTHS を書き換える関数
# ============================================================

def set_months(months: int):
    """10_generate_history.py の MONTHS と fetch_start を書き換える"""
    gen_path = BASE_DIR / "10_generate_history.py"
    text = gen_path.read_text(encoding="utf-8")

    import re

    # MONTHS = XX を書き換え
    text = re.sub(r"^MONTHS\s*=\s*\d+",
                  f"MONTHS = {months}",
                  text, flags=re.MULTILINE)

    # fetch_start の days を書き換え（余裕を持たせる）
    extra = 200
    days  = months * 30 + extra
    text  = re.sub(
        r"fetch_start\s*=\s*\(today\s*-\s*timedelta\(days=\d+\)\)",
        f"fetch_start = (today - timedelta(days={days}))",
        text
    )

    gen_path.write_text(text, encoding="utf-8")
    print(f"  MONTHS={months} に設定しました")


# ============================================================
# スクリプトを実行して出力を取得する関数
# ============================================================

def run_script(script_name: str, label: str) -> str:
    """スクリプトを実行して標準出力を文字列で返す"""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        return f"[ERROR] {script_name} が見つかりません\n"

    def decode_output(data: bytes | None) -> str:
        if not data:
            return ""
        for enc in ("utf-8", "cp932"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    try:
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        result = subprocess.run(
            [PYTHON, str(script_path)],
            capture_output=True,
            text=False,
            cwd=str(BASE_DIR),
            env=child_env,
            timeout=600  # 10分タイムアウト
        )
        stdout_text = decode_output(result.stdout)
        stderr_text = decode_output(result.stderr)

        if result.returncode != 0:
            return f"[ERROR] {label}\n{stderr_text}\n"
        return stdout_text

    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] {label} がタイムアウトしました\n"
    except Exception as e:
        return f"[ERROR] {label}: {e}\n"


# ============================================================
# メイン処理
# ============================================================

def main():
    print("=" * 60)
    print("  全スパン一括分析")
    print(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for years, months, label in SPANS:
        print(f"\n{'='*60}")
        print(f"  [{label}] {years}年分（{months}ヶ月）の分析開始")
        print(f"{'='*60}")

        # 10年分は既存データを使う（再生成しない）
        if months == 120:
            print("  10年分は既存データを使用します（スキップ）")
        else:
            # MONTHS を書き換えてデータ生成
            print(f"\n  [1/6] データ生成中...")
            set_months(months)
            gen_output = run_script("10_generate_history.py", "データ生成")
            # 最後の数行だけ表示
            for line in gen_output.strip().split("\n")[-5:]:
                print(f"    {line}")

        # 各分析を実行して結果を収集
        all_output = []
        all_output.append("=" * 65)
        all_output.append(f"  {label.upper()} 全スパン分析レポート")
        all_output.append(f"  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        all_output.append("=" * 65)

        for i, (script, desc) in enumerate(ANALYSIS_SCRIPTS, 2):
            print(f"\n  [{i}/6] {desc} 実行中...")
            output = run_script(script, desc)

            # コンソールにも進捗表示
            lines = output.strip().split("\n") if output else []
            print(f"    → {len(lines)}行の出力")

            all_output.append(f"\n{'#'*65}")
            all_output.append(f"# {desc}")
            all_output.append(f"{'#'*65}")
            all_output.append(output)

        # テキストファイルに保存
        output_path = RESULT_DIR / f"{label}_analysis.txt"
        output_text = "\n".join(all_output)
        output_path.write_text(output_text, encoding="utf-8")

        print(f"\n  💾 {output_path.name} に保存しました")
        print(f"     ({len(output_text):,} 文字)")

    # ============================================================
    # 完了後に MONTHS を 10年に戻す
    # ============================================================
    print("\n\n10_generate_history.py を10年設定に戻します...")
    set_months(120)

    print("\n" + "=" * 60)
    print("  全スパン分析 完了！")
    print("=" * 60)
    print(f"\n保存先: {RESULT_DIR}")
    for years, months, label in SPANS:
        path = RESULT_DIR / f"{label}_analysis.txt"
        size = path.stat().st_size / 1024 if path.exists() else 0
        print(f"  {label}_analysis.txt  ({size:.1f} KB)")

    print(f"\n次のステップ:")
    print(f"  各テキストファイルをOpusに貼り付けて最終フィルター設計へ")


if __name__ == "__main__":
    main()