# 01. nanoclawとは？ 🦅

## 一言で言うと

> **nanoclaw** は、Telegram（などのチャットサービス）と Claude AI を繋ぐ「橋渡しシステム」です。

---

## なぜ nanoclaw が必要なのか？

Claude は Anthropic のサーバー上で動く AI です。
Telegram はメッセージアプリです。
この2つは**そのままでは繋がりません**。

```
Telegram  ←→  ???  ←→  Claude API
```

nanoclaw がその「???」の部分を担います。

```
Telegram  ←→  nanoclaw  ←→  Claude API
```

---

## nanoclaw ができること

### 1. チャットへの応答
- Telegram のメッセージを受け取り、Claude に送り、返答を返す

### 2. スケジュールタスク
- 「毎朝9時に天気を教えて」のような定期実行タスクを管理

### 3. グループ管理
- 複数のグループ・チャットを登録して管理できる

### 4. Claude Code として動作
- ただ会話するだけでなく、**ファイル操作・コード実行・Web検索** なども実行できる

---

## "nanoclaw" という名前について

- **nano** = 小さい・軽量
- **claw** = 爪（鷹の爪 → 掴む・繋ぐイメージ）

小さくて、しっかり掴む（繋ぐ）システム、というイメージです 🦅

---

## 対応しているチャットサービス

| サービス | 状況 |
|----------|------|
| Telegram | ✅ 対応 |
| WhatsApp | ✅ 対応 |
| Discord  | ✅ 対応 |

---

## このリポジトリの場所について

```
Docker コンテナ内:
  /workspace/group/nanoclaw-learning/  ← ここ（永続領域）

ホストマシン上:
  nanoclaw の group フォルダ内に保存されている

GitHub:
  push すれば github.com/yuzuponikemi/nanoclaw-learning で見られる
```

---

## 次のステップ

→ [02. アーキテクチャ](02-architecture.md) でシステム全体の構成を見てみましょう
