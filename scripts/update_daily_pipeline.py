"""
update_daily_pipeline.py
日次パイプライン更新スクリプト（GitHub Actions で毎朝実行）

INPUT:
  - yfinance から US / JP ETF の OHLC データを取得
  - data/processed/signals.csv, trades.csv, performance.csv（既存データ）

OUTPUT:
  - data/raw/us_etf.csv, data/raw/jp_etf.csv を更新
  - data/processed/signals.csv, trades.csv, performance.csv に新行を追記

使い方:
  python scripts/update_daily_pipeline.py
  （git push は GitHub Actions の workflow が担う）
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

# ================================================================
# 定数
# ================================================================

L = 60          # ローリングウィンドウ
K = 3           # 主成分数
K0 = 3          # 事前部分空間の次元
LAMBDA = 0.9    # 正則化パラメータ
TOP_N = 3       # ロング / ショート業種数

FETCH_START = "2010-01-01"
CFULL_START = "2010-01-01"
CFULL_END   = "2014-12-31"

INITIAL_CAPITAL = 1_000_000

US_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
JP_TICKERS = ["1617.T","1618.T","1619.T","1620.T","1621.T","1622.T","1623.T",
              "1624.T","1625.T","1626.T","1627.T","1628.T","1629.T","1630.T",
              "1631.T","1632.T","1633.T"]

JP_KEYS = {
    "1617.T": "food",        "1618.T": "energy",
    "1619.T": "construction","1620.T": "materials",
    "1621.T": "pharma",      "1622.T": "auto",
    "1623.T": "steel",       "1624.T": "machinery",
    "1625.T": "electronics", "1626.T": "it_services",
    "1627.T": "utilities",   "1628.T": "transport",
    "1629.T": "trading",     "1630.T": "retail",
    "1631.T": "banks",       "1632.T": "finance",
    "1633.T": "realestate",
}

US_CYCLICAL  = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL  = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]

WEEKDAY_EN = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

N_US = len(US_TICKERS)
N_JP = len(JP_TICKERS)
N = N_US + N_JP


# ================================================================
# PCA SUB 関数群（archive/05.py / 10_generate_history.py から）
# ================================================================

def standardize(df: pd.DataFrame):
    mu    = df.mean()
    sigma = df.std().replace(0, np.nan)
    z     = (df - mu) / sigma
    return z, mu, sigma


def build_prior_subspace() -> np.ndarray:
    """固定V0（28×3）を作る。全データ期間を通じて不変。"""
    v1 = np.ones(N) / np.sqrt(N)

    v2_raw = np.zeros(N)
    v2_raw[:N_US] = +1.0
    v2_raw[N_US:] = -1.0
    v2_raw -= np.dot(v2_raw, v1) * v1
    v2 = v2_raw / np.linalg.norm(v2_raw)

    all_tickers = US_TICKERS + JP_TICKERS
    v3_raw = np.zeros(N)
    for i, t in enumerate(all_tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = +1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0
    v3_raw -= np.dot(v3_raw, v1) * v1
    v3_raw -= np.dot(v3_raw, v2) * v2
    norm3 = np.linalg.norm(v3_raw)
    v3 = v3_raw / norm3 if norm3 > 1e-10 else np.zeros(N)

    return np.column_stack([v1, v2, v3])


def build_C0(V0: np.ndarray, cfull: np.ndarray) -> np.ndarray:
    D0     = np.diag(np.diag(V0.T @ cfull @ V0))
    C0_raw = V0 @ D0 @ V0.T
    diag_v = np.where(np.diag(C0_raw) > 0, np.diag(C0_raw), 1.0)
    C0     = np.diag(1.0 / np.sqrt(diag_v)) @ C0_raw @ np.diag(1.0 / np.sqrt(diag_v))
    np.fill_diagonal(C0, 1.0)
    return C0


def calc_pca_sub_signal(
    window_us: pd.DataFrame,
    window_jp: pd.DataFrame,
    z_us_today: pd.Series,
    C0: np.ndarray,
    V0: np.ndarray,
) -> pd.Series:
    """PCA SUB シグナルを計算する（17業種のシグナル値を返す）。"""
    window_us = window_us.reindex(columns=US_TICKERS)
    window_jp = window_jp.reindex(columns=JP_TICKERS)
    combined  = pd.concat([window_us, window_jp], axis=1).fillna(0.0)

    if len(combined) < K + 1:
        return pd.Series(np.nan, index=JP_TICKERS)

    z_comb, _, _ = standardize(combined)
    z_comb = z_comb.fillna(0.0)
    Ct = z_comb.values.T @ z_comb.values / len(z_comb)

    C_reg = (1 - LAMBDA) * Ct + LAMBDA * C0

    evals, evecs = np.linalg.eigh(C_reg)
    idx = np.argsort(evals)[::-1]
    V = evecs[:, idx[:K]]       # (N, K)

    V_us = V[:N_US, :]
    V_jp = V[N_US:, :]

    _, mu_us, sigma_us = standardize(window_us)
    sigma_us = sigma_us.replace(0, np.nan)
    z_today = (z_us_today - mu_us) / sigma_us
    z_today = z_today.fillna(0)

    ft = V_us.T @ z_today.values
    signal = V_jp @ ft

    return pd.Series(signal, index=JP_TICKERS)


# ================================================================
# 1. ETF データ取得
# ================================================================

def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[1] ETFデータ取得中...")
    us_raw = yf.download(US_TICKERS, start=FETCH_START, auto_adjust=True, progress=False)
    jp_raw = yf.download(JP_TICKERS, start=FETCH_START, auto_adjust=True, progress=False)
    us_raw[["Open","Close"]].to_csv(RAW_DIR / "us_etf.csv")
    jp_raw[["Open","Close"]].to_csv(RAW_DIR / "jp_etf.csv")
    print(f"  US: {len(us_raw)} 日, JP: {len(jp_raw)} 日")
    return us_raw, jp_raw


# ================================================================
# 2. リターン計算
# ================================================================

def calc_returns(us_raw: pd.DataFrame, jp_raw: pd.DataFrame):
    us_cc = us_raw["Close"].reindex(columns=US_TICKERS).pct_change()
    jp_cc = jp_raw["Close"].reindex(columns=JP_TICKERS).pct_change()
    jp_oc = ((jp_raw["Close"] - jp_raw["Open"]) / jp_raw["Open"]).reindex(columns=JP_TICKERS)

    common = us_cc.index.intersection(jp_cc.index).intersection(jp_oc.index)
    us_cc  = us_cc.loc[common]
    jp_cc  = jp_cc.loc[common]
    jp_oc  = jp_oc.loc[common]
    return us_cc, jp_cc, jp_oc, common


# ================================================================
# 3. Cfull・C0 計算
# ================================================================

def build_cfull_matrix(us_cc: pd.DataFrame, jp_cc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mask = (us_cc.index >= CFULL_START) & (us_cc.index <= CFULL_END)
    us_c = us_cc.loc[mask].reindex(columns=US_TICKERS)
    jp_c = jp_cc.loc[mask].reindex(columns=JP_TICKERS)
    combined = pd.concat([us_c, jp_c], axis=1).fillna(0.0)
    z_c, _, _ = standardize(combined)
    z_c = z_c.fillna(0.0)
    cfull = z_c.values.T @ z_c.values / len(z_c)
    V0 = build_prior_subspace()
    C0 = build_C0(V0, cfull)
    return C0, V0


# ================================================================
# 4. 既存データ読み込み・状態取得
# ================================================================

def load_existing_state() -> dict:
    """既存のCSVから継続状態を読み込む。"""
    sig_path  = PROCESSED_DIR / "signals.csv"
    trd_path  = PROCESSED_DIR / "trades.csv"
    perf_path = PROCESSED_DIR / "performance.csv"

    state = {
        "last_date": None,
        "capital": float(INITIAL_CAPITAL),
        "capital_long": float(INITIAL_CAPITAL),
        "total_days": 0,
        "wins": 0,
        "sig_columns": None,
        "recent_20": [],
    }

    if sig_path.exists():
        df = pd.read_csv(sig_path, parse_dates=["date"])
        if len(df) > 0:
            state["last_date"] = df["date"].max()
            state["sig_columns"] = df.columns.tolist()

    if trd_path.exists():
        df = pd.read_csv(trd_path)
        if len(df) > 0:
            # 直近20日分の is_correct を引き継ぐ（win_rate_20d 計算用）
            state["recent_20"] = df["is_correct"].iloc[-20:].tolist()

    if perf_path.exists():
        df = pd.read_csv(perf_path)
        if len(df) > 0:
            last = df.iloc[-1]
            state["capital"]      = float(last["capital"])
            state["capital_long"] = float(last["capital_long_only"])
            state["total_days"]   = int(last["total_days"])
            state["wins"]         = round(float(last["win_rate_total"]) * state["total_days"])

    if state["last_date"] is not None:
        print(f"  既存データ最終日: {state['last_date'].date()}")
    else:
        print("  既存データなし（全期間を生成します）")

    return state


# ================================================================
# 5. 新規行を生成して既存CSVに追記
# ================================================================

def append_new_rows(
    common: pd.Index,
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    C0: np.ndarray,
    V0: np.ndarray,
    state: dict,
) -> int:
    last_date = state["last_date"]
    capital      = state["capital"]
    capital_long = state["capital_long"]
    total_days   = state["total_days"]
    wins         = state["wins"]
    sig_columns  = state["sig_columns"]
    # 既存データの直近 is_correct を引き継いで win_rate_20d を正確に計算する
    recent_20_buf: list[int] = list(state.get("recent_20", []))

    sig_path  = PROCESSED_DIR / "signals.csv"
    trd_path  = PROCESSED_DIR / "trades.csv"
    perf_path = PROCESSED_DIR / "performance.csv"

    sig_rows  = []
    trd_rows  = []
    perf_rows = []

    # 追記対象: last_date より後の common_date を next_date とする
    for k in range(1, len(common)):
        next_date   = common[k]
        target_date = common[k - 1]

        # last_date 以前はスキップ
        if last_date is not None and next_date <= last_date:
            continue

        # ウィンドウ確保チェック
        if k < L:
            continue

        # jp_oc に欠損がないか確認（市場未終了の場合は NaN になる）
        oc_row = jp_oc.loc[next_date]
        if oc_row.isna().all():
            print(f"  {next_date.date()}: jp_oc が空 → スキップ（未確定）")
            continue

        # シグナル計算
        window_us  = us_cc.iloc[k - L : k]
        window_jp  = jp_cc.iloc[k - L : k]
        z_us_today = us_cc.loc[target_date]

        signal = calc_pca_sub_signal(window_us, window_jp, z_us_today, C0, V0)

        ranked    = signal.dropna().sort_values(ascending=False)
        long_t    = ranked.head(TOP_N).index.tolist()
        short_t   = ranked.tail(TOP_N).index.tolist()

        sig_strength = float(ranked.iloc[0] - ranked.iloc[-1]) if len(ranked) > 0 else np.nan
        sig_spread   = float(ranked.iloc[:TOP_N].mean() - ranked.iloc[-TOP_N:].mean()) if len(ranked) >= TOP_N * 2 else np.nan

        # リターン計算
        long_ret  = float(oc_row[long_t].mean())  if long_t  else np.nan
        short_ret = float(oc_row[short_t].mean()) if short_t else np.nan
        strat_ret = long_ret - short_ret if not (np.isnan(long_ret) or np.isnan(short_ret)) else np.nan
        is_correct = int(long_ret > short_ret) if not np.isnan(strat_ret) else 0

        if np.isnan(strat_ret):
            print(f"  {next_date.date()}: strat_ret が NaN → スキップ")
            continue

        date_str = next_date.strftime("%Y-%m-%d")
        wday_en  = WEEKDAY_EN[next_date.dayofweek]
        month    = next_date.month

        # 累積
        capital      *= (1 + strat_ret)
        capital_long *= (1 + long_ret)
        total_days   += 1
        wins         += is_correct
        recent_20_buf.append(is_correct)
        if len(recent_20_buf) > 20:
            recent_20_buf.pop(0)

        # -- signals 行 --
        sig_row: dict = {
            "date":             date_str,
            "signal_strength":  round(sig_strength, 6),
            "signal_spread":    round(sig_spread,   6),
        }
        for t in JP_TICKERS:
            sig_row[f"signal_{JP_KEYS[t]}"] = round(float(signal.get(t, np.nan)), 6)
        for rank, t in enumerate(ranked.index, 1):
            sig_row[f"rank_{JP_KEYS[t]}"] = rank
        sig_rows.append(sig_row)

        # -- trades 行 --
        trd_rows.append({
            "date":             date_str,
            "weekday":          wday_en,
            "month":            month,
            "long_1":           JP_KEYS.get(long_t[0],  "") if len(long_t)  > 0 else "",
            "long_2":           JP_KEYS.get(long_t[1],  "") if len(long_t)  > 1 else "",
            "long_3":           JP_KEYS.get(long_t[2],  "") if len(long_t)  > 2 else "",
            "short_1":          JP_KEYS.get(short_t[0], "") if len(short_t) > 0 else "",
            "short_2":          JP_KEYS.get(short_t[1], "") if len(short_t) > 1 else "",
            "short_3":          JP_KEYS.get(short_t[2], "") if len(short_t) > 2 else "",
            "signal_strength":  round(sig_strength, 6),
            "signal_spread":    round(sig_spread,   6),
            "long_return":      round(long_ret,     6),
            "short_return":     round(short_ret,    6),
            "strategy_return":  round(strat_ret,    6),
            "long_only_return": round(long_ret,     6),
            "is_correct":       is_correct,
        })

        # -- performance 行 --
        win_rate_20d = sum(recent_20_buf) / len(recent_20_buf) if recent_20_buf else 0.0
        perf_rows.append({
            "date":              date_str,
            "strategy_return":   round(strat_ret,  6),
            "long_return":       round(long_ret,   6),
            "short_return":      round(short_ret,  6),
            "long_only_return":  round(long_ret,   6),
            "capital":           round(capital,    0),
            "capital_long_only": round(capital_long, 0),
            "pnl":               round(capital - INITIAL_CAPITAL, 0),
            "pnl_pct":           round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL, 6),
            "is_correct":        is_correct,
            "win_rate_total":    round(wins / total_days, 4),
            "win_rate_20d":      round(win_rate_20d, 4),
            "total_days":        total_days,
        })

    n_new = len(sig_rows)
    if n_new == 0:
        print("  追加行なし（データは最新の状態です）")
        return 0

    # CSV に追記
    def append_csv(path: Path, rows: list[dict], col_order: list | None = None) -> None:
        df_new = pd.DataFrame(rows)
        if col_order:
            # 既存カラム順に合わせる（存在しない列は NaN）
            df_new = df_new.reindex(columns=col_order)
        if path.exists():
            df_new.to_csv(path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(path, index=False)

    append_csv(sig_path,  sig_rows,  sig_columns)
    append_csv(trd_path,  trd_rows)
    append_csv(perf_path, perf_rows)

    print(f"  {n_new} 件を追記しました（最新: {sig_rows[-1]['date']}）")
    return n_new


# ================================================================
# メイン
# ================================================================

def main() -> None:
    print("=" * 60)
    print("LeadLag 日次パイプライン更新")
    print("=" * 60)

    us_raw, jp_raw = fetch_data()

    print("[2] リターン計算中...")
    us_cc, jp_cc, jp_oc, common = calc_returns(us_raw, jp_raw)
    print(f"  共通営業日: {len(common)} 日 ({common[0].date()} ~ {common[-1].date()})")

    print("[3] C0 / V0 構築中...")
    C0, V0 = build_cfull_matrix(us_cc, jp_cc)
    print(f"  Cfull期間: {CFULL_START} ~ {CFULL_END}")

    print("[4] 既存データ読み込み中...")
    state = load_existing_state()

    print("[5] 新規行を計算・追記中...")
    n_added = append_new_rows(common, us_cc, jp_cc, jp_oc, C0, V0, state)

    print("=" * 60)
    if n_added > 0:
        print(f"完了: {n_added} 日分のデータを追記しました")
    else:
        print("完了: 新規データなし")
    print("=" * 60)


if __name__ == "__main__":
    main()
