# llmpg: LLM/Agent Playground

ローカルLLMの活用・RAG・ファインチューニング・Agentパターンを実践的に学ぶプレイグラウンドです。
Ollama（`http://host.docker.internal:11434`）がDockerコンテナから直接利用可能な環境を活かした構成になっています。

## 📚 カリキュラム

| ノートブック | テーマ | 前提知識 |
|---|---|---|
| [01_ollama_basics](notebooks/01_ollama_basics.ipynb) | ローカルLLMをAPIで呼ぶ・モデル比較 | なし |
| [02_prompt_engineering](notebooks/02_prompt_engineering.ipynb) | プロンプト設計・few-shot・CoT | 01完了 |
| [03_rag_from_scratch](notebooks/03_rag_from_scratch.ipynb) | RAGをゼロから実装（埋め込み→検索→生成） | 01完了 |
| [04_finetuning_with_unsloth](notebooks/04_finetuning_with_unsloth.ipynb) | LoRAファインチューニング（GPU/Colab向け） | 01, 02完了 |
| [05_llm_evaluation](notebooks/05_llm_evaluation.ipynb) | ベンチマーク・LLM-as-judge自動評価 | 01, 02完了 |
| [06_agent_patterns](notebooks/06_agent_patterns.ipynb) | Tool use / ReAct / マルチエージェントパターン | 01〜03完了 |

## 🏗️ ディレクトリ構成

```
llmpg/
├── notebooks/
│   ├── 01_ollama_basics.ipynb
│   ├── 02_prompt_engineering.ipynb
│   ├── 03_rag_from_scratch.ipynb
│   ├── 04_finetuning_with_unsloth.ipynb
│   ├── 05_llm_evaluation.ipynb
│   └── 06_agent_patterns.ipynb
├── src/
│   ├── __init__.py
│   ├── ollama_client.py     # Ollama APIラッパー
│   └── rag_utils.py         # 埋め込み・ベクトル検索ユーティリティ
├── requirements.txt
└── README.md
```

## 🚀 クイックスタート

### 1. 依存パッケージのインストール

```bash
cd llmpg
pip install -r requirements.txt
```

### 2. Ollamaのセットアップ

Ollamaが起動していることを確認し、必要なモデルをプルします：

```bash
# デフォルトモデル（チャット）
ollama pull qwen2.5:7b

# 埋め込みモデル（03_rag用）
ollama pull nomic-embed-text
```

### 3. Jupyterの起動

```bash
jupyter lab
```

ブラウザで `notebooks/01_ollama_basics.ipynb` から始めてください。

## 🛠️ src モジュール

### `ollama_client.py` - Ollama APIラッパー

```python
from src.ollama_client import OllamaClient, get_client

# クライアント作成
client = get_client(model="qwen2.5:7b")

# 接続確認
print(client.is_available())  # True/False

# テキスト生成
response = client.generate("日本の首都は？")
print(response)

# チャット（多ターン）
history = []
response = client.chat("こんにちは！", history=history)
history.append({"role": "user", "content": "こんにちは！"})
history.append({"role": "assistant", "content": response})

# 埋め込み
vector = client.embed("東京は日本の首都です", model="nomic-embed-text")
print(len(vector))  # 768

# ストリーミング
for chunk in client.generate("俳句を作って", stream=True):
    print(chunk, end="", flush=True)
```

### `rag_utils.py` - RAGユーティリティ

```python
from src.rag_utils import RAGPipeline, VectorStore, split_text
from src.ollama_client import get_client

client = get_client()

# RAGパイプライン（簡単な使い方）
pipeline = RAGPipeline(client)
pipeline.add_texts([
    "東京は日本の首都で、人口約1400万人の大都市です。",
    "大阪は関西地方の中心都市で、たこ焼きが有名です。",
])
answer = pipeline.query("東京の人口はどれくらいですか？")
print(answer)

# テキスト分割
chunks = split_text("長いテキスト...", chunk_size=200, chunk_overlap=20)
```

## 📝 各ノートブックの概要

### 01. Ollama基礎
- Ollamaへの接続確認とモデル一覧取得
- `generate` / `chat` / `embed` API の使い方
- ストリーミング出力
- モデル比較ベンチマーク

### 02. プロンプトエンジニアリング
- ゼロショット・フューショットの違い
- Chain-of-Thought（CoT）推論
- システムプロンプトによるペルソナ設定
- 再利用可能なプロンプトテンプレート

### 03. RAGをゼロから実装
- 埋め込みベクトルの直感的理解
- コサイン類似度による意味検索
- `VectorStore` と `RAGPipeline` の使い方
- ファイルベースのQ&Aシステム

### 04. LoRAファインチューニング（GPU必須）
- Full Fine-tuning vs. LoRAの違い
- Unslothによる効率的な学習
- Alpacaフォーマットのデータセット作成
- GGUFエクスポートとOllamaへのインポート

### 05. LLM評価
- 基本評価指標（完全一致、単語オーバーラップ）
- LLM-as-Judge（LLMで別LLMを評価）
- A/Bテストによる比較評価
- ハルシネーション検出

### 06. エージェントパターン
- Tool Use（関数呼び出し）
- ReActパターン（思考→行動→観察）
- 会話メモリと状態管理
- マルチエージェント（研究者 + 執筆者）

## 🔧 技術スタック

| カテゴリ | ツール |
|---|---|
| 推論 | Ollama（OpenAI互換API） |
| デフォルトモデル | `qwen2.5:7b` |
| 埋め込み | `nomic-embed-text`（Ollama経由） |
| ベクトルDB | インメモリ（VectorStore）、ChromaDB（オプション） |
| ファインチューニング | Unsloth + TRL |

## 🔗 関連リンク

- [Ollama 公式サイト](https://ollama.ai/)
- [Ollama API ドキュメント](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [ChromaDB ドキュメント](https://docs.trychroma.com/)
