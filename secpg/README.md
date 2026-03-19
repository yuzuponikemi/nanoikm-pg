# secpg — Security Playground

> **完全初心者向け** サイバーセキュリティ学習教材

「SQLインジェクションって何？」というところから始めて、
実際に手を動かしながら攻撃の仕組みと防御を学びます。

---

## ⚠️ 重要な注意事項

この教材は **教育・防御目的のみ** です。

- 練習は必ず **ローカル環境の labs/ アプリ** で行う
- 他人のシステムに無断でテストすることは **違法** です
- 学んだ知識は「守る側」として活用してください

---

## 📚 カリキュラム

| # | ノートブック | テーマ | 難易度 |
|---|-------------|--------|--------|
| 01 | [セキュリティとは](notebooks/01_intro_to_security.ipynb) | 脅威・脆弱性・攻撃者の分類 | ⭐ |
| 02 | [Webの仕組み](notebooks/02_web_basics.ipynb) | HTTP・リクエスト・レスポンス・Cookie | ⭐ |
| 03 | [SQLインジェクション](notebooks/03_sql_injection.ipynb) | 最も有名な脆弱性を手を動かして理解 | ⭐⭐ |
| 04 | [XSS](notebooks/04_xss.ipynb) | クロスサイトスクリプティング | ⭐⭐ |
| 05 | [認証の脆弱性](notebooks/05_authentication.ipynb) | パスワード・セッション・JWT | ⭐⭐⭐ |
| 06 | [ネットワークセキュリティ](notebooks/06_network_security.ipynb) | パケット・ポートスキャン・暗号化 | ⭐⭐⭐ |
| 07 | [CTF入門](notebooks/07_ctf_intro.ipynb) | Capture The Flag で実力試し | ⭐⭐⭐ |
| 08 | [セキュアコーディング](notebooks/08_secure_coding.ipynb) | 脆弱性を作らないコードの書き方 | ⭐⭐ |

---

## 🧪 ラボ環境

```
labs/
└── vulnerable_app/   ← 意図的に脆弱なWebアプリ（練習専用）
    ├── app.py
    └── templates/
```

### 起動方法

```bash
cd labs/vulnerable_app
pip install flask
python app.py
# http://localhost:5000 にアクセス
```

---

## 学習の流れ

```
ノートブックで概念を理解
        ↓
labs/ のアプリで実際に攻撃を試す
        ↓
防御コードを書いて修正
        ↓
次のトピックへ
```

---

## 前提知識

- Pythonの基本（変数・関数・ループ）
- Webブラウザの使い方
- 以上！セキュリティの知識は不要です
