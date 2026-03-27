"""
RAG ユーティリティ
==================
Retrieval-Augmented Generation (RAG) を構築するための
埋め込み・ベクトル検索・チャンク分割ユーティリティを提供します。

依存: numpy, ollama_client
オプション: chromadb (ベクトルDB使用時)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ollama_client import OllamaClient

# デフォルト設定
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


# ------------------------------------------------------------------
# データクラス
# ------------------------------------------------------------------


@dataclass
class Document:
    """テキストドキュメントとメタデータのコンテナ。"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(content='{preview}...', metadata={self.metadata})"


@dataclass
class RetrievalResult:
    """検索結果（ドキュメントとスコア）。"""

    document: Document
    score: float

    def __repr__(self) -> str:
        preview = self.document.content[:60].replace("\n", " ")
        return f"RetrievalResult(score={self.score:.4f}, content='{preview}...')"


# ------------------------------------------------------------------
# テキスト分割
# ------------------------------------------------------------------


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separator: str = "\n\n",
) -> list[str]:
    """テキストを重複付きチャンクに分割する。

    Parameters
    ----------
    text : str
        分割するテキスト。
    chunk_size : int
        各チャンクの最大文字数。
    chunk_overlap : int
        隣接チャンク間の重複文字数。
    separator : str
        最初に試みる分割区切り文字。

    Returns
    -------
    list[str]
        分割されたテキストチャンクのリスト。

    Examples
    --------
    >>> chunks = split_text("長いテキスト...", chunk_size=100, chunk_overlap=20)
    >>> len(chunks)
    5
    """
    # まず段落で分割を試みる
    paragraphs = text.split(separator)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + len(separator) <= chunk_size:
            current = (current + separator + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # パラグラフ自体がchunk_sizeより大きい場合は文単位で分割
            if len(para) > chunk_size:
                sub_chunks = _split_by_sentences(para, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""
            else:
                current = para

    if current:
        chunks.append(current)

    # オーバーラップを適用
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        return overlapped

    return chunks


def _split_by_sentences(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """文単位でテキストを分割するヘルパー。"""
    sentences = re.split(r"(?<=[。．.!?！？])\s*", text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks or [text]


# ------------------------------------------------------------------
# 埋め込み生成
# ------------------------------------------------------------------


def embed_documents(
    documents: list[Document],
    client: OllamaClient,
    embed_model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
) -> list[Document]:
    """ドキュメントリストに埋め込みベクトルを付与する。

    Parameters
    ----------
    documents : list[Document]
        埋め込みを追加するドキュメントリスト。
    client : OllamaClient
        Ollamaクライアント。
    embed_model : str
        埋め込みモデル名。
    batch_size : int
        バッチサイズ（一度にAPIに送るテキスト数）。

    Returns
    -------
    list[Document]
        埋め込みが付与されたドキュメントリスト（元のリストを変更して返す）。
    """
    texts = [doc.content for doc in documents]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = client.embed(batch, model=embed_model)
        for j, emb in enumerate(embeddings):
            documents[i + j].embedding = emb

    return documents


# ------------------------------------------------------------------
# 類似度計算
# ------------------------------------------------------------------


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """2つのベクトルのコサイン類似度を計算する。

    Returns
    -------
    float
        類似度スコア（-1.0〜1.0、高いほど類似）。
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ------------------------------------------------------------------
# インメモリベクトルストア
# ------------------------------------------------------------------


class VectorStore:
    """インメモリのシンプルなベクトルストア。

    小規模な RAG 実験に最適。大規模なデータには ChromaDB を使用してください。

    Examples
    --------
    >>> store = VectorStore()
    >>> store.add_documents(docs)
    >>> results = store.search(query_embedding, top_k=3)
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []

    def __len__(self) -> int:
        return len(self._documents)

    def add_documents(self, documents: list[Document]) -> None:
        """埋め込み済みドキュメントをストアに追加する。"""
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(
                    f"Document '{doc.content[:30]}...' に埋め込みがありません。"
                    " embed_documents() を先に呼び出してください。"
                )
        self._documents.extend(documents)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """クエリ埋め込みに最も近いドキュメントを返す。

        Parameters
        ----------
        query_embedding : list[float]
            クエリの埋め込みベクトル。
        top_k : int
            返す結果の最大数。
        min_score : float
            最低スコアフィルタ（0.0〜1.0）。

        Returns
        -------
        list[RetrievalResult]
            スコア降順の検索結果リスト。
        """
        if not self._documents:
            return []

        scores = [
            cosine_similarity(query_embedding, doc.embedding)  # type: ignore[arg-type]
            for doc in self._documents
        ]

        results = [
            RetrievalResult(document=doc, score=score)
            for doc, score in zip(self._documents, scores)
            if score >= min_score
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """ストア内のすべてのドキュメントを削除する。"""
        self._documents.clear()


# ------------------------------------------------------------------
# RAGパイプライン
# ------------------------------------------------------------------


class RAGPipeline:
    """テキストチャンク→埋め込み→検索→生成の一連のパイプライン。

    Examples
    --------
    >>> pipeline = RAGPipeline(client)
    >>> pipeline.add_texts(["東京は日本の首都です。", "大阪は関西の中心地です。"])
    >>> answer = pipeline.query("東京について教えてください")
    >>> print(answer)
    """

    def __init__(
        self,
        client: OllamaClient,
        embed_model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = 3,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.system_prompt = system_prompt or (
            "あなたは役立つアシスタントです。"
            "提供されたコンテキストのみを使用して質問に答えてください。"
            "コンテキストに情報がない場合は「情報が見つかりません」と答えてください。"
        )
        self.store = VectorStore()

    def add_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        """テキストリストをチャンク分割・埋め込み・追加する。

        Parameters
        ----------
        texts : list[str]
            追加するテキストリスト。
        metadata : list[dict] | None
            各テキストに対応するメタデータ（省略可）。
        """
        if metadata is None:
            metadata = [{} for _ in texts]

        documents: list[Document] = []
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            chunks = split_text(text, self.chunk_size, self.chunk_overlap)
            for j, chunk in enumerate(chunks):
                doc_meta = {**meta, "source_index": i, "chunk_index": j}
                documents.append(Document(content=chunk, metadata=doc_meta))

        embed_documents(documents, self.client, self.embed_model)
        self.store.add_documents(documents)

    def add_file(self, file_path: str | Path, **metadata: Any) -> None:
        """テキストファイルを読み込んで追加する。"""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        self.add_texts([text], metadata=[{"source": str(path), **metadata}])

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """クエリに関連するドキュメントを検索する。"""
        query_emb = self.client.embed(query, model=self.embed_model)
        return self.store.search(query_emb, top_k=top_k or self.top_k)

    def query(self, question: str, top_k: int | None = None) -> str:
        """質問に対してRAGで回答を生成する。

        Parameters
        ----------
        question : str
            ユーザーの質問。
        top_k : int | None
            取得するチャンク数。

        Returns
        -------
        str
            生成された回答。
        """
        results = self.retrieve(question, top_k=top_k)

        if not results:
            context = "（関連するコンテキストが見つかりませんでした）"
        else:
            context_parts = []
            for i, r in enumerate(results, 1):
                context_parts.append(f"[コンテキスト {i}]\n{r.document.content}")
            context = "\n\n".join(context_parts)

        prompt = f"""以下のコンテキストを参照して質問に答えてください。

{context}

質問: {question}
回答:"""

        return self.client.chat(
            message=prompt,
            system=self.system_prompt,
            temperature=0.3,
        )

    @property
    def num_documents(self) -> int:
        """ストア内のチャンク数を返す。"""
        return len(self.store)
