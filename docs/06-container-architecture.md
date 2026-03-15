# 06. コンテナアーキテクチャ詳解 🐳

> ソースコード (`container-runner.ts`, `agent-runner/src/index.ts`) を実際に読んで確認した内容です。

---

## よくある誤解と正解

| 誤解 ❌ | 正解 ✅ |
|--------|--------|
| Agent ツールを呼ぶたびに新コンテナが作られる | **グループ（チャット）ごと**に新コンテナが作られる |
| サブエージェントは別コンテナで動く | サブエージェントは**同コンテナ内**で動く |

---

## コンテナが作られるタイミング

```
✅ Telegram グループがメッセージを送ったとき
✅ スケジュールタスクが実行されたとき
❌ Agent ツール（サブエージェント）が呼ばれたとき ← 作られない
```

---

## 全体構造

```
ホストマシン（nanoclaw プロセスが動いている）
│
├── グループAのメッセージ受信
│       ↓
│   Docker コンテナ起動: nanoclaw-groupA-{timestamp}
│       │
│       ├── メインエージェント (Claude Code)
│       │       ├── Agent ツール → サブエージェントA（同コンテナ内・別プロセス）
│       │       └── Agent ツール → サブエージェントB（同コンテナ内・別プロセス）
│       │
│       ├── IPC ポーリング（次のメッセージを 500ms ごとに監視）
│       │       ↓ 次のメッセージが来たら
│       │       そのまま同コンテナで継続処理
│       │       ↓ _close sentinel 受信 or タイムアウト
│       └── コンテナ終了
│
└── グループBのメッセージ → 完全に独立した別コンテナ
```

---

## なぜ Agent ツールは同コンテナなのか

`agent-runner/src/index.ts` のコードに答えがあります：

```typescript
// settings.json に書き込まれる設定
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"  // ← これがキー！
  }
}

// allowedTools にエージェントチーム用ツールが含まれる
allowedTools: [
  'Task', 'TaskOutput', 'TaskStop',
  'TeamCreate', 'TeamDelete', 'SendMessage',
  ...
]
```

Claude Code SDK の **"agent teams" 機能** を使っており、
サブエージェントは SDK レベルで同プロセス内に生成されます。

**理由**: 新コンテナ起動には数秒かかるため、
サブエージェントのたびに作ると遅すぎる。
セキュリティ境界はグループ間で確保すれば十分。

---

## コンテナのライフサイクル

```
1. メッセージ受信
      ↓
2. docker run -i --rm nanoclaw-{name}-{timestamp}
      ↓
3. stdin に ContainerInput (JSON) を送信
      ↓
4. エージェントが処理・応答
      ↓
5. stdout に OUTPUT_START_MARKER / OUTPUT_END_MARKER で結果を出力
      ↓
6. IPC ポーリングで次のメッセージを待機（500ms ごと）
      ↓
7a. 次のメッセージ → 同コンテナで継続
7b. _close sentinel → コンテナ終了
7c. タイムアウト → コンテナ強制終了
```

---

## グループ間のセキュリティ隔離

コンテナのマウント構成がグループによって異なります：

| マウント先 | メイングループ | その他グループ |
|-----------|--------------|--------------|
| `/workspace/project` | ✅ 読み取り専用 | ❌ なし |
| `/workspace/group` | ✅ 自グループのフォルダ | ✅ 自グループのフォルダのみ |
| `/workspace/global` | ❌ なし | ✅ 読み取り専用 |
| `/workspace/ipc` | ✅ 自グループ専用 | ✅ 自グループ専用 |

→ グループAのコンテナからグループBのデータは**原理的に見えない**。

---

## `.env` のマスキング

```typescript
// container-runner.ts より
mounts.push({
  hostPath: '/dev/null',
  containerPath: '/workspace/project/.env',
  readonly: true,
});
```

`/dev/null` を `.env` の場所に**上書きマウント**することで、
ホストの API キー等がコンテナから見えないようにしています。
API 認証は別途 **credential proxy** 経由で行われます。

---

## 実演で確認したこと（2026-03-15）

Agent ツールでサブエージェントを起動した際：

```
メインエージェントのホスト名:  3cd55a0d80d0
サブエージェントのホスト名:    3cd55a0d80d0  ← 同じ！
```

→ 同一コンテナ内で動いていることをライブで確認。

---

## 次のステップ

→ [07. IPC（プロセス間通信）](07-ipc.md) で
コンテナとホスト間のメッセージのやり取りを見てみましょう
