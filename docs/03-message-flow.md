# 03. メッセージフロー詳細 💬

## あなたが「こんにちは」と送ったとき、何が起きているか

### ステップ1: 送信

```
[あなた] ──「こんにちは」──▶ [Telegram アプリ]
```

Telegram アプリがメッセージをサーバーに送信します。

---

### ステップ2: Bot API が受信

```
[Telegram サーバー] ──── Webhook ────▶ [nanoclaw]

送信されるデータ例 (JSON):
{
  "update_id": 12345678,
  "message": {
    "message_id": 42,
    "from": {
      "id": 987654321,
      "first_name": "Yusuke"
    },
    "chat": {
      "id": -1001234567890,   ← グループID
      "type": "group"
    },
    "text": "こんにちは"
  }
}
```

---

### ステップ3: nanoclaw がメッセージを処理

```
nanoclaw 内部の処理:

1. グループID (-1001234567890) で登録グループか確認
      ↓
2. トリガーワード確認（例: "@Claude" が含まれるか）
      ↓
3. 会話履歴を取得（コンテキスト管理）
      ↓
4. システムプロンプト + 履歴 + 新メッセージを構築
```

---

### ステップ4: Anthropic API へリクエスト

```
nanoclaw ────HTTP POST────▶ api.anthropic.com

送信データ例:
{
  "model": "claude-sonnet-4-5",
  "messages": [
    {
      "role": "user",
      "content": "こんにちは"
    }
  ],
  "tools": [
    // Claude Code として動くためのツール定義
    { "name": "Bash", ... },
    { "name": "Read", ... },
    { "name": "Write", ... },
    ...
  ]
}
```

---

### ステップ5: Claude が応答を生成

```
[Anthropic サーバー内]

入力: "こんにちは"
  ↓
Claude が処理
  ↓
出力: "こんにちは！何かお手伝いできることはありますか？"
```

---

### ステップ6: 応答を返す

```
api.anthropic.com ────▶ nanoclaw ────▶ Telegram Bot API ────▶ あなた

応答データ例:
{
  "content": [
    {
      "type": "text",
      "text": "こんにちは！何かお手伝いできることはありますか？"
    }
  ]
}
```

---

## ツール使用時のフロー（Claude Code の場合）

Claude Code の場合、単純な応答とは異なり「ツール」を使うことがあります。

```
ユーザー: "現在のディレクトリにあるファイルを教えて"

[Claude]
  → "Bash ツールを使おう" と判断
  ↓
[nanoclaw]
  → ls コマンドを実行
  → 結果を Claude に返す
  ↓
[Claude]
  → 結果を解釈して自然文で返答
  ↓
[ユーザー]
  ← "以下のファイルがあります: ..."
```

### ツール使用の往復図

```
Claude ──[ツール呼び出し要求]──▶ nanoclaw
Claude ◀──[ツール実行結果]────── nanoclaw
Claude ──[最終応答]────────────▶ nanoclaw ──▶ Telegram
```

この往復が**1ターンの中で複数回**起きることもあります。

---

## コンテキスト（会話履歴）の管理

nanoclaw は会話の文脈を維持するために履歴を管理しています：

```
グループ A の履歴:
  [user]: "こんにちは"
  [assistant]: "こんにちは！..."
  [user]: "あなたについて教えて"
  [assistant]: "私はClaudeです..."
  [user]: "なるほど"  ← 今ここ
```

これにより「さっきの話の続き」ができます。

---

## 次のステップ

→ [04. コンポーネント解説](04-components.md) で各部品の詳細を見てみましょう
