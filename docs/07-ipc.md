# 07. IPC（プロセス間通信）の仕組み 📡

> コンテナとホスト（nanoclaw）がどうやってメッセージをやり取りするかを解説します。

---

## IPC とは

**IPC = Inter-Process Communication（プロセス間通信）**

コンテナ内のエージェントと、ホスト上の nanoclaw プロセスが
情報をやり取りするための仕組みです。

---

## 通信の方向

```
ホスト (nanoclaw)
    │
    │  ① stdin に ContainerInput JSON を送る
    ▼
Docker コンテナ (agent-runner)
    │
    │  ② stdout に OUTPUT_MARKER で結果を返す
    ▼
ホスト (nanoclaw)
    │
    │  ③ /workspace/ipc/input/ にファイルを置く（追加メッセージ）
    ▼
Docker コンテナ（500ms ごとにポーリングして拾う）
    │
    │  ④ /workspace/ipc/input/_close を置く（終了シグナル）
    ▼
コンテナ終了
```

---

## 詳細：各通信チャネル

### ① 起動時：stdin → ContainerInput

```json
{
  "prompt": "ユーザーのメッセージ",
  "sessionId": "e80ed078-...",
  "groupFolder": "telegram_main",
  "chatJid": "tg:-1001234567890",
  "isMain": true,
  "isScheduledTask": false,
  "assistantName": "Claude"
}
```

コンテナは stdin を EOF まで読み、JSON をパースして処理開始。

---

### ② 結果出力：stdout → OUTPUT_MARKER

```
---NANOCLAW_OUTPUT_START---
{"status":"success","result":"こんにちは！...","newSessionId":"abc-123"}
---NANOCLAW_OUTPUT_END---
```

センチネルマーカーで囲むことで、stdout のノイズと区別して確実にパース。
**ストリーミング**対応で、複数回出力される場合もある（agent teams）。

---

### ③ 追加メッセージ：IPC input ディレクトリ

```
/workspace/ipc/input/
    └── {timestamp}.json   ← ホストがここにファイルを置く

ファイルの中身:
{
  "type": "message",
  "text": "続きのメッセージ"
}
```

コンテナ内のエージェントは **500ms ごとにポーリング**して、
新しいファイルがあれば読み取り → 削除 → Claude に渡す。

これにより、**コンテナを再起動せずに会話を継続**できる。

---

### ④ 終了シグナル：_close sentinel

```
/workspace/ipc/input/_close  ← このファイルを置くだけ
```

ポーリング時に `_close` を検出したら：
1. ファイルを削除
2. MessageStream を終了
3. コンテナが自然に exit

---

## セッション管理

コンテナをまたいで会話を継続するために `sessionId` を使います：

```
1回目の会話
  コンテナ起動 → sessionId なし → Claude が新セッション作成
  → newSessionId: "abc-123" を返す → ホストが保存

2回目の会話
  コンテナ起動 → sessionId: "abc-123" を渡す → Claude が前の会話を記憶した状態で再開
```

---

## /workspace/ipc の構成

```
/workspace/ipc/
├── available_groups.json   # 登録グループ一覧（register_group ツール用）
├── current_tasks.json      # 実行中スケジュールタスク一覧
├── input/                  # 追加メッセージ投入口
│   └── {timestamp}.json    # ホストが置く・エージェントが消費する
├── messages/               # エージェント→ホスト のメッセージ
└── tasks/                  # タスク管理
```

---

## IPC を使う nanoclaw ツール（エージェント側）

| ツール | 内部動作 |
|--------|---------|
| `schedule_task` | `/workspace/ipc/` 経由でホストにタスク登録を依頼 |
| `list_tasks` | `current_tasks.json` を読む |
| `send_message` | `/workspace/ipc/messages/` にファイルを書く |
| `register_group` | `available_groups.json` を参照して登録 |

---

## まとめ図

```
nanoclaw（ホスト）                コンテナ（エージェント）
─────────────────────────────────────────────────────
stdin → ContainerInput JSON  →  受信・パース・処理開始
                              ←  stdout: OUTPUT_MARKER×N
ipc/input/{msg}.json         →  500ms ポーリングで受信
ipc/input/_close             →  終了を検知・exit
─────────────────────────────────────────────────────
共有ファイルシステム (/workspace/ipc, /workspace/group)
で双方向に状態を共有
```
