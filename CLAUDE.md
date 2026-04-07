# nanoikm-pg — nanoclaw 学習リポジトリ

nanoclaw（Telegram × Claude Code）の仕組みを図解と実践で学ぶための教材集。

## パス

- プロジェクトルート: `/workspace/group/nanoikm-pg/`
- 現在の状況: `PROJECT_STATUS.md`

## ディレクトリ構成

```
nanoikm-pg/
├── docs/                  ← メインドキュメント（01〜07）
│   ├── 01-overview.md     ← nanoclawとは何か
│   ├── 02-architecture.md ← システム構成図
│   ├── 03-message-flow.md ← メッセージが届くまでの流れ
│   ├── 04-components.md   ← 各パーツの役割
│   ├── 05-hands-on.md     ← ハンズオン
│   ├── 06-container-architecture.md ← コンテナの仕組み
│   └── 07-ipc.md          ← コンテナとホストのIPC通信
├── mlpg/                  ← Machine Learning Playground（別プロジェクト）
│   └── CLAUDE.md          ← mlpg固有のルール
├── debatepg/              ← ディベート練習（サブプロジェクト）
├── rlpg/                  ← 強化学習 Playground
├── secpg/                 ← セキュリティ Playground
├── statspg/               ← 統計学 Playground
└── README.md              ← 目次
```

## 学習コンテンツの対象範囲

- nanoclawがTelegramとClaudeをどう繋ぐか
- メッセージ処理とルーティングの仕組み
- スケジュールタスク・グループ管理
- Claude Codeとして動くとはどういうことか
- コンテナの起動タイミングと仕組み（ソースコード読解）
- サブエージェントの動作場所
- コンテナ↔ホスト間のIPC通信

## サブプロジェクト: mlpg（機械学習プレイグラウンド）

機械学習の基礎から応用まで学ぶ Jupyter ノートブック集（118以上）。
独自のルールは `/workspace/group/nanoikm-pg/mlpg/CLAUDE.md` を参照。

重要ルール:
- 新しいノートブック追加・変更後は `SYLLABUS.md` を必ず更新
- 図のラベル・タイトルはすべて英語（japanize_matplotlib は禁止）
- ノートブック作成は `NOTEBOOK_GUIDELINES.md` に従う
