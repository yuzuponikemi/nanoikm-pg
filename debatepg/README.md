# debatepg — マルチエージェント議論シミュレーター

複数のAIペルソナが任意のテーマについて議論し、モデレーターが最終結論を出すCLIツール。

## ペルソナ

| | 名前 | 立場 |
|--|------|------|
| 🐂 | Bull | 楽観主義者・成長重視 |
| 🐻 | Bear | 懐疑主義者・リスク重視 |
| 🔧 | Engineer | 技術実現性派 |
| 📊 | Analyst | データ・エビデンス重視 |
| 🎯 | CEO | 戦略家・ROI重視 |

## 議論フロー

```
[Moderator] テーマ設定
[Round 1]   各ペルソナ: 冒頭主張
[Round 2]   各ペルソナ: 反論・補足
[Round 3]   各ペルソナ: 最終立場
[Moderator] 全発言統合 → 推奨アクション
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 基本（Ollamaデフォルト）

```bash
python main.py "RustをAIプロジェクトに採用すべきか"
```

### オプション

```bash
# ラウンド数変更
python main.py "LLMのファインチューニングは必要か" --rounds 4

# ペルソナ絞り込み
python main.py "Kubernetes導入の是非" --personas bull,bear,engineer

# 結果を保存
python main.py "AIエージェントに意思決定を委ねるべきか" --save

# Claude APIを使用
python main.py "テーマ" --backend claude

# カスタムモデル
python main.py "テーマ" --model qwen3.5:latest

# Ollamaホスト指定（ローカル直接接続など）
python main.py "テーマ" --ollama-host http://localhost:11434

# ペルソナ一覧表示
python main.py --list-personas
```

### 環境変数

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `DEBATE_MODEL` | 使用モデル名 | `llama3.2:latest` (ollama) / `claude-3-5-haiku-20241022` (claude) |
| `ANTHROPIC_API_KEY` | Claude API キー | — |

## バックエンド

| バックエンド | 説明 | デフォルトモデル |
|------------|------|----------------|
| `ollama` | ローカルLLM（無料・プライベート） | `llama3.2:latest` |
| `claude` | Anthropic Claude API | `claude-3-5-haiku-20241022` |
