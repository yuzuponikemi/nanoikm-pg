# 🦅 nanoclaw 学習リポジトリ

> Telegram × Claude Code を繋ぐ **nanoclaw** の仕組みを図解と実践で学ぶ教材集

---

## 📚 目次

| # | ドキュメント | 内容 |
|---|-------------|------|
| 01 | [概要・全体像](docs/01-overview.md) | nanoclawとは何か |
| 02 | [アーキテクチャ](docs/02-architecture.md) | システム構成の図解 |
| 03 | [メッセージフロー](docs/03-message-flow.md) | メッセージが届くまでの流れ |
| 04 | [コンポーネント解説](docs/04-components.md) | 各パーツの役割と仕組み |
| 05 | [ハンズオン](docs/05-hands-on.md) | 実際に試してみよう |

---

## 🗺️ 全体像（ざっくり）

```
あなた (Telegram)
     ↕
[Telegram Bot API]
     ↕
[nanoclaw サーバー]  ← 中継・管理
     ↕
[Claude API (Anthropic)]  ← AI の頭脳
```

---

## 🚀 この教材で学べること

- ✅ nanoclaw がどうやって Telegram と Claude を繋ぐか
- ✅ メッセージがやり取りされる仕組み
- ✅ スケジュールタスク・グループ管理の概念
- ✅ Claude Code として動くとはどういうことか

---

## 🛠️ 前提知識

- Telegram の基本操作
- API とは何かの概念（なんとなくでOK）
- JSON の読み方（なんとなくでOK）

---

*最終更新: 2026-03-15*
