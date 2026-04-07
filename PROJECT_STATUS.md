# nanoikm-pg — 現在の状況

最終更新: 2026-04-07

## 状態

🟢 稼働中（教材作成・拡充フェーズ）

## プロジェクト概要

nanoclaw の仕組みを学ぶための教材リポジトリ。メインドキュメント 07 本 + 複数のサブプロジェクト（ML/RL/セキュリティ/統計/LLM/ディベート）を収録。

## 直近の作業

- [2026-04-07] 全サブプロジェクトの状態調査・PROJECT_STATUS.md 更新
- [2026-03-15] docs/06-container-architecture.md、docs/07-ipc.md を追加（コンテナ・IPC詳解）

## サブプロジェクト状態

| サブプロジェクト | 状態 | 内容 | 進行状況 |
|---|---|---|---|
| `docs/` (01〜07) | 🟢 完成 | nanoclawアーキテクチャドキュメント7本 | — |
| `mlpg/` | 🟢 稼働中 | 機械学習ノートブック（153本）。ルール: `mlpg/CLAUDE.md` | 定期メンテナンス中 |
| `llmpg/` | 🟢 稼働中 | LLM/Agent Playground（6本のノートブック：Ollama基礎→RAG→ファインチューニング→Agent） | アクティブ |
| `rlpg/` | 🟢 進行中 | 強化学習プレイグラウンド（11本のノートブック + 3つのPR進行中） | DQN/PPO/SAC/DDPGの実装PR |
| `secpg/` | 🟢 完成 | セキュリティ学習教材（完全初心者向け、カリキュラム完成） | 実装済み(issue#17) |
| `statspg/` | 🟢 稼働中 | 統計検定準1級学習（33本のノートブック、全32章対応） | メンテナンス中 |
| `debatepg/` | 🟡 設計段階 | マルチエージェント議論シミュレーター（5種類のペルソナ）| 実装検討中 |

## オープンPR状況

| # | タイトル | ブランチ | 状態 |
|---|---|---|---|
| #33 | [rlpg] DDPG実装ノートブック (notebook 13) | issue-32-ddpg-notebook | MERGEABLE |
| #31 | [rlpg] SAC実装ノートブック (notebook 12) | issue-30-sac-notebook | MERGEABLE |
| #28 | [rlpg] PPO実装ノートブック (notebook 11) | issue-27-ppo-notebook | MERGEABLE |

## 次のステップ

- [ ] rlpg の3つのオープンPR（PPO/SAC/DDPG）をレビュー・マージ
- [ ] debatepg の実装開始（ペルソナ・議論フロー・CLI実装）
- [ ] 新しい教材があれば docs/08 以降を作成

## 現在の課題・メモ

- mlpgのノートブック追加時はSYLLABUS.mdの更新を忘れずに
- rlpgのノートブック実装は継続中（強化学習アルゴリズムの網羅的カバレッジ）
- debatepg は設計は完了だがCLI実装はまだ — 次のアクティビティタスク候補
