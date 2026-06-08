"""
=================================================================
MAE分析用 OHLCデータ取得（JQuants API Key直接認証版）
=================================================================
実行: python scripts/20_download_ohlc.py
出力: data/sector_ohlc.csv

.env に JQUANTS_API_KEY を設定
=================================================================
"""

import os
import sys
import requests
import pandas as pd
import time
from dotenv import load_dotenv

BASE_DIR = r"C:\Users\hg317\Desktop\projects\leadlag"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "sector_ohlc.csv")

load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("JQUANTS_API_KEY")

if not API_KEY:
    print("✗ JQUANTS_API_KEY が .env にありません")
    sys.exit(1)

BASE_URL = "https://api.jquants.com/v1"

SECTORS = {
    "16170": "food",
    "16180": "energy",
    "16190": "construction",
    "16200": "materials",
    "16210": "pharma",
    "16220": "auto",
    "16230": "steel",
    "16240": "machinery",
    "16250": "electronics",
    "16260": "it_services",
    "16270": "utilities",
    "16280": "transport",
    "16290": "trading",
    "16300": "retail",
    "16310": "banks",
    "16320": "finance",
    "16330": "realestate",
}


def get_id_token():
    """API Key → リフレッシュトークン → IDトークン（複数の認証方法を試す）"""

    # === 方法1: API Keyをそのまま apiKey フィールドで送る ===
    print("  方法1: apiKey認証...", end=" ", flush=True)
    resp = requests.post(
        f"{BASE_URL}/token/auth_user",
        headers={"Content-Type": "application/json"},
        json={"apiKey": API_KEY},
        timeout=30,
    )
    if resp.status_code == 200 and "refreshToken" in resp.json():
        refresh_token = resp.json()["refreshToken"]
        print("✓")
        return _refresh_to_id(refresh_token)

    print(f"({resp.status_code})")

    # === 方法2: API Keyをリフレッシュトークンとして直接使う ===
    print("  方法2: リフレッシュトークン直接...", end=" ", flush=True)
    resp2 = requests.post(
        f"{BASE_URL}/token/auth_refresh?refreshtoken={API_KEY}",
        timeout=30,
    )
    if resp2.status_code == 200 and "idToken" in resp2.json():
        print("✓")
        return resp2.json()["idToken"]

    print(f"({resp2.status_code})")

    # === 方法3: API KeyをAuthorizationヘッダーで送る ===
    print("  方法3: Bearerヘッダー...", end=" ", flush=True)
    resp3 = requests.get(
        f"{BASE_URL}/prices/daily_quotes?code=16170&from=2026-03-01&to=2026-03-19",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    if resp3.status_code == 200:
        print("✓ (API Keyで直接アクセス可能)")
        return API_KEY  # API Key自体がトークンとして使える

    print(f"({resp3.status_code})")

    # === 方法4: X-API-KEY ヘッダー ===
    print("  方法4: X-API-KEYヘッダー...", end=" ", flush=True)
    resp4 = requests.get(
        f"{BASE_URL}/prices/daily_quotes?code=16170&from=2026-03-01&to=2026-03-19",
        headers={"X-API-KEY": API_KEY},
        timeout=30,
    )
    if resp4.status_code == 200:
        print("✓ (X-API-KEYで直接アクセス可能)")
        return "__XAPIKEY__"

    print(f"({resp4.status_code})")

    # 全方法失敗
    print("\n✗ 全ての認証方法が失敗しました")
    print(f"  API_KEY先頭: {API_KEY[:15]}...")
    print(f"  方法1レスポンス: {resp.text[:200]}")
    print(f"  方法2レスポンス: {resp2.text[:200]}")
    print(f"  方法3レスポンス: {resp3.text[:200]}")
    print(f"  方法4レスポンス: {resp4.text[:200]}")
    print()
    print("  → JQuantsダッシュボードでパスワードを確認し、")
    print("    .env に以下を追加してください:")
    print("    JQUANTS_MAIL_ADDRESS=あなたのメールアドレス")
    print("    JQUANTS_PASSWORD=あなたのパスワード")
    sys.exit(1)


def _refresh_to_id(refresh_token):
    """リフレッシュトークン → IDトークン"""
    print("  IDトークン取得...", end=" ", flush=True)
    resp = requests.post(
        f"{BASE_URL}/token/auth_refresh?refreshtoken={refresh_token}",
        timeout=30,
    )
    if resp.status_code == 200 and "idToken" in resp.json():
        print("✓")
        return resp.json()["idToken"]
    print(f"✗ ({resp.status_code})")
    sys.exit(1)


def fetch_quotes(token, code, from_date, to_date, use_xapi=False):
    """日足OHLC取得"""
    if use_xapi:
        headers = {"X-API-KEY": token}
    else:
        headers = {"Authorization": f"Bearer {token}"}

    all_rows = []
    params = {"code": code, "from": from_date, "to": to_date}

    while True:
        resp = requests.get(
            f"{BASE_URL}/prices/daily_quotes",
            headers=headers, params=params, timeout=30,
        )
        if resp.status_code == 401:
            return None
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        all_rows.extend(data.get("daily_quotes", []))

        pagination_key = data.get("pagination_key")
        if pagination_key:
            params["pagination_key"] = pagination_key
            time.sleep(0.3)
        else:
            break

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def main():
    print("=" * 60)
    print("  OHLCデータ取得 — JQuants API")
    print("=" * 60)

    print("\n認証中（複数方法を順番に試します）...")
    token = get_id_token()
    use_xapi = (token == "__XAPIKEY__")
    if use_xapi:
        token = API_KEY
    print()

    all_data = []
    failed = []

    for code, name in SECTORS.items():
        print(f"  {code} ({name})...", end=" ", flush=True)

        df = fetch_quotes(token, code, "2016-05-01", "2026-03-20", use_xapi)

        if df is None:
            print("トークン再取得...", end=" ", flush=True)
            token = get_id_token()
            df = fetch_quotes(token, code, "2016-05-01", "2026-03-20", use_xapi)

        if df is None or len(df) == 0:
            print("⚠ データなし")
            failed.append(code)
            continue

        if len(all_data) == 0:
            print(f"\n    [DEBUG] カラム: {list(df.columns)}")

        # カラム名マッピング（調整済み価格を優先）
        rename = {}
        for c in df.columns:
            cl = c.lower().replace("_", "")
            if cl == "date": rename[c] = "date"
            elif "adjustmentopen" in cl or cl == "adjustmentopen": rename[c] = "open"
            elif "adjustmenthigh" in cl or cl == "adjustmenthigh": rename[c] = "high"
            elif "adjustmentlow" in cl or cl == "adjustmentlow": rename[c] = "low"
            elif "adjustmentclose" in cl or cl == "adjustmentclose": rename[c] = "close"

        for target in ["open", "high", "low", "close"]:
            if target not in rename.values():
                for c in df.columns:
                    if c.lower() == target:
                        rename[c] = target

        df = df.rename(columns=rename)
        needed = ["date", "open", "high", "low", "close"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"⚠ 列不足: {missing} 利用可能: {list(df.columns)[:15]}")
            failed.append(code)
            continue

        df = df[needed].copy()
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values("date").reset_index(drop=True)
        df["sector"] = name
        df["ticker"] = code[:4]

        all_data.append(df)
        print(f"✓ {len(df)}行")
        time.sleep(0.5)

    if not all_data:
        print("\n✗ データ取得失敗")
        sys.exit(1)

    result = pd.concat(all_data, ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print(f"  ✓ 保存: {OUTPUT_PATH}")
    print(f"    行数: {len(result):,}")
    print(f"    業種: {result['sector'].nunique()}/17")
    if failed:
        print(f"    失敗: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()