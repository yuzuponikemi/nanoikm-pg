# nanoikm-pg — プロジェクト概要

## このリポジトリについて

このリポジトリは **体系的な機械学習・強化学習の学習教材** を開発するプロジェクトです。

最終目標: 初学者から中級者が、概念理解 → 実装 → 可視化 → 比較 という流れで学べる、完成度の高いJupyterノートブック集を作ること。

## 開発の場

このグループ（WhatsApp/Telegram）で開発を進めていきます。Andyがissueを拾って実装・PR作成を担当します。

## リポジトリ構成

```
nanoikm-pg/
├── rlpg/         # 強化学習 (Reinforcement Learning Playground)
├── mlpg/         # 機械学習 (Machine Learning Playground)
├── statspg/      # 統計学 (Statistics Playground)
└── docs/
```

## 各モジュールの方針

### rlpg
- 環境: 倒立振子 (InvertedPendulumEnv)
- ポリシー: `src/policies/` に各アルゴリズムを実装
- ノートブック: 01_〜 の連番で、概念 → 実装 → 可視化 → 比較の流れ
- 実装済み: RandomPolicy, LinearPolicy, NeuralNetworkPolicy, QPolicy

### mlpg
- 教師あり学習・深層学習の実装ノートブック
- Transformer, VAEなど高度なアーキテクチャも含む予定

### statspg
- 統計量とMLモデルの橋渡し

## 開発フロー

1. GitHub issueから未着手のものを選ぶ
2. サブエージェントに設計を考えさせる
3. メインエージェントが実装してPRを作成
4. Yusukeがレビュー・マージ

## コーディング方針

- ノートブックは **教育的コメント** を豊富に入れる（概念説明 → コード → 可視化）
- 既存コードのスタイル・インタフェースを踏襲する
- `Policy` 基底クラスのインタフェースを厳守 (`get_action`, `get_params`, `set_params`)
- 返り値の辞書キーは既存関数と統一する
