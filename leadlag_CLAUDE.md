# Lead-Lag Strategy — AI 作業ガイド

このファイルはすべての AI が**必ず最初に読む**設計書。
作業前に必ず確認し、禁止事項・分岐ルールに従って進めること。

---

## プロジェクト概要

米国セクターETF（US）の前日終値リターン（CC）で
翌日の日本個別株（OC: 寄り→引け）を予測するロング・ショート戦略。

- 手法: PCA SUB（部分空間正則化PCA）
- λ=0.9、K=3、L=60日ウィンドウ（論文値、変更禁止）
- Cfull 推定期間: 2010-01-01〜2014-12-31（変更禁止）
- ロング上位3業種 / ショート下位3業種

---

## 現在のファイル状態

```
data/processed/
  signals.csv          ETF版シグナル (2021-04〜2026-03)  ← 短期のみ
  trades.csv           ETF版売買記録 (2021-04〜)
  performance.csv      ETF版累積成績 (2021-04〜)
  signals_v2.csv       個別株版シグナル (2010-04〜2026-03)  ← Phase2 で作成済み
  trades_v2.csv        個別株版売買記録
  performance_v2.csv   個別株版累積成績

output/reports/
  phase1_ic_ranking.csv  ← 無効（signals.csv の 2021〜しか使っていない）
```

### ⚠️ 現在の問題点（必ず把握すること）

1. **Phase 1 の IC テストが無効**
   - signals.csv は 2021-04〜の 1,157 日しかない
   - IC テストに使った期間が短すぎて信頼性がない
   - **フル期間（2010年〜）のシグナルで再テストが必要**

2. **Phase 2 と ETF 版の比較期間が異なる**
   - 個別株版: 2010-04〜（3,791 日）
   - ETF 版: 2021-04〜（1,157 日）
   - **同期間で比較しないと意味がない**

---

## 正しい作業フロー

```
Step 0: ETF版フルシグナル計算（2010〜）
          → データ基盤として先に作る

Step 1: Phase 1 再実行（フル期間 IC テスト）
          → ETF版フルシグナル × 候補株 OC の相関を計算
          → 期間を 3 分割して安定性を確認
          → 業種ごとに分岐判定（採用A/B/C/D）

Step 2: Phase 2 再実行（必要な業種のみ）
          → 採用A/B 業種のみで個別株版バックテスト

Step 3: Phase 3（公正比較）
          → 同期間で ETF版 vs 個別株版 を比較

Step 4: 採用モデル決定 → 本番パイプライン更新
```

---

## Step 0: ETF 版フルシグナル計算

**スクリプト**: `scripts/step0_calc_full_signals.py`（未作成）

### 処理内容
1. yfinance で US ETF + JP ETF を 2010-01-01 から取得
2. PCA SUB を全期間で実行
3. `data/processed/signals_etf_full.csv` に保存

### 出力カラム（signals.csv と同じ形式）
```
date, signal_strength, signal_spread,
signal_food, signal_energy, ..., signal_realestate,
rank_food, rank_energy, ..., rank_realestate
```

### 完了条件
- `signals_etf_full.csv` の期間が 2010-04〜現在（約3,800日）

---

## Step 1: Phase 1（フル期間 IC テスト）

**スクリプト**: `scripts/phase1_ic_test.py`（再作成が必要）

### IC（情報係数）の定義
```
IC_sector_t = Corr(signal_sector[t],  r_stock[t+1]^OC)
```

シグナルの入力は `signals_etf_full.csv` を使うこと（2021〜の短期 signals.csv は使わない）。

### 候補株リスト（phase1_ic_test.py の CANDIDATES に定義）

| 業種 | 候補ティッカー |
|------|-------------|
| food | 2802, 2503, 2502, 2914 |
| energy | 5020, 5019, 1605 |
| construction | 1801, 1802, 1803, 5233 |
| materials | 4063, 4188, 3407, 4452 |
| pharma | 4502, 4519, 4568, 4523 |
| auto | 7203, 7267, 7269, 7201 |
| steel | 5401, 5713, 5802 |
| machinery | 6367, 6302, 6273, 7011 |
| electronics | 6758, 6861, 6702, 6971 |
| it_services | 9984, 9432, 9433, 4307 |
| utilities | 9501, 9502, 9503, 9531 |
| transport | 9020, 9101, 9104, 9107 |
| trading | 8058, 8031, 8053, 8001 |
| retail | 9983, 3382, 8267, 3092 |
| banks | 8306, 8316, 8411 |
| finance | 8766, 8725, 8604, 8750 |
| realestate | 8801, 8802, 8830 |

### 計算する指標（全指標が必須）

| 指標 | 説明 | 採用基準 |
|------|------|---------|
| IC_full | 全期間の IC | 主指標 |
| IC_2010_2015 | 前期 IC | 安定性確認 |
| IC_2016_2020 | 中期 IC | 安定性確認 |
| IC_2021_now | 後期 IC | 安定性確認 |
| n_days | 有効サンプル数 | ≥ 750 必須 |
| positive_periods | 3 期間中プラスの数 | ≥ 2 必須 |
| win_rate | 翌日 OC が正の割合 | 参考 |

### 業種ごとの分岐判定ルール

**採用 A（そのまま確定）**
```
条件:
  IC_full >= 0.05
  n_days >= 750
  positive_periods >= 2（3期間中2期間以上でICがプラス）

アクション:
  当該業種の1位銘柄を確定採用
```

**採用 B（条件付き採用）**
```
条件:
  0.02 <= IC_full < 0.05
  n_days >= 750
  positive_periods >= 2

アクション:
  1位と2位を Phase 2 で並行バックテストして比較
```

**再探索 C（候補拡張）**
```
条件:
  IC_full < 0.02 または n_days < 750

アクション:
  当該業種だけ候補を 3〜5 銘柄追加して Phase 1 を再実行
  → 追加候補は phase1_ic_test.py の CANDIDATES を更新する
```

**逆張り候補 D（符号反転テスト）**
```
条件:
  IC_full <= -0.03 が 3 期間すべてで安定継続

アクション:
  シグナル符号反転ケースを別実験として記録
  本線モデルには採用しない（過学習防止）
```

### ポートフォリオ全体の分岐

**P1（採用 A が 14 業種以上）**
```
アクション: Phase 2 へ進む（単一構成）
```

**P2（採用 A + B が 14 業種以上、A 単独は不足）**
```
アクション: 曖昧業種のみ複数案を作成して Phase 2 で比較
```

**P3（採用 A + B が 14 業種未満）**
```
アクション: 候補ユニバース再設計を先に実施。Phase 2 へは進まない。
```

### 出力ファイル
```
output/reports/phase1_ic_ranking.csv
  カラム: sector, ticker, IC_full, IC_2010_2015, IC_2016_2020, IC_2021_now,
          n_days, positive_periods, win_rate, decision（A/B/C/D）
```

---

## Step 2: Phase 2（個別株版バックテスト）

**スクリプト**: `scripts/phase2_full_backtest.py`

### 実行条件
- Step 1 の `phase1_ic_ranking.csv` が完成していること
- ポートフォリオ分岐が P1 または P2 であること

### 処理内容
1. `phase1_ic_ranking.csv` から採用銘柄を読み込む
2. 各銘柄の OHLC を 2010-01-01 から取得
3. JP CC（Close-to-Close）で PCA SUB を再計算
4. JP OC（Open-to-Close）でリターンを評価
5. signals_v2.csv / trades_v2.csv / performance_v2.csv に保存

### ⚠️ 注意
- signals.csv / trades.csv / performance.csv は上書きしない
- v2 系ファイルに保存すること

---

## Step 3: Phase 3（公正比較）

**スクリプト**: `scripts/phase3_etf_full_backtest.py`

### 処理内容
1. ETF 版を 2010〜現在でフルバックテスト
2. signals_etf_full.csv / trades_etf_full.csv / performance_etf_full.csv に保存
3. 同期間で ETF 版 vs 個別株版を比較

### 採用判定

```
個別株版採用:
  Sharpe_v2 >= Sharpe_etf × 0.95
  かつ流動性フィルタ通過

ETF版維持（ペーパートレード前提）:
  Sharpe_etf が有意に高い（× 1.1 以上）
  ただし実運用コスト控除後は優位が消えるため実運用しない

ハイブリッド:
  業種ごとに優位モデルが異なる場合
  流動性が高い業種のみ個別株、その他は除外
```

---

## PCA SUB 固定パラメータ（変更禁止）

| パラメータ | 値 |
|-----------|-----|
| λ (LAMBDA) | 0.9 |
| K（主成分数） | 3 |
| L（ウィンドウ） | 60 日 |
| Cfull 期間 | 2010-01-01〜2014-12-31 |
| V0 構成 | グローバル / 国スプレッド / シクリカル |

---

## AIへの禁止事項

1. **signals.csv（2021〜）を IC テストのシグナル源に使わない**
   → 必ず `signals_etf_full.csv`（2010〜）を使う

2. **期間の違うデータを直接比較しない**
   → 比較するときは必ず同期間に揃える

3. **ETF版と個別株版の CSV を混在させない**
   → ETF版: signals.csv / trades.csv / performance.csv
   → 個別株版: signals_v2.csv / trades_v2.csv / performance_v2.csv

4. **PCA SUB の固定パラメータを変更しない**

5. **archive/ 以下のファイルを本番に戻さない**

6. **Phase 1 が未完了のまま Phase 2 に進まない**

---

## 現在の作業キュー（AIが見るべき次のアクション）

```
[ ] Step 0: scripts/step0_calc_full_signals.py を作成・実行
      → data/processed/signals_etf_full.csv を生成する

[ ] Step 1: scripts/phase1_ic_test.py を書き直す
      → signals_etf_full.csv を入力として使う
      → 3期間分割の安定性チェックを実装する
      → 業種ごとに A/B/C/D の判定を出力する

[ ] Step 2: Phase 2 は Step 1 完了後に実行

[ ] Step 3: Phase 3 は Step 2 完了後に実行
```
