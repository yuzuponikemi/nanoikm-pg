"""
Ollama API ラッパー
====================
Ollama (http://host.docker.internal:11434) への統一アクセスを提供します。
OpenAI互換エンドポイントとネイティブOllama APIの両方をサポートします。
"""

from __future__ import annotations

import json
import time
from typing import Any, Generator, Iterator

import requests

# デフォルト設定
DEFAULT_BASE_URL = "http://host.docker.internal:11434"
DEFAULT_MODEL = "qwen2.5:7b"


class OllamaClient:
    """Ollama APIへの統一アクセスクライアント。

    Examples
    --------
    >>> client = OllamaClient()
    >>> response = client.chat("こんにちは！")
    >>> print(response)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # 接続確認
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Ollamaサーバーが起動しているか確認する。"""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """利用可能なモデル一覧を返す。"""
        resp = self._session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    # ------------------------------------------------------------------
    # テキスト生成
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """プロンプトからテキストを生成する（/api/generate エンドポイント）。

        Parameters
        ----------
        prompt : str
            入力プロンプト。
        model : str | None
            使用するモデル名。None の場合 self.model を使用。
        system : str | None
            システムプロンプト。
        temperature : float
            サンプリング温度 (0.0〜2.0)。
        max_tokens : int
            最大トークン数。
        stream : bool
            True の場合ストリーミングジェネレータを返す。

        Returns
        -------
        str | Iterator[str]
            stream=False なら完全なレスポンス文字列、stream=True なら文字列のイテレータ。
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }
        if system:
            payload["system"] = system

        resp = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
            stream=stream,
        )
        resp.raise_for_status()

        if stream:
            return self._stream_generate(resp)
        return resp.json()["response"]

    def _stream_generate(self, resp: requests.Response) -> Generator[str, None, None]:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")
                if data.get("done"):
                    break

    # ------------------------------------------------------------------
    # チャット
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """チャット形式でメッセージを送信する（/api/chat エンドポイント）。

        Parameters
        ----------
        message : str
            ユーザーメッセージ。
        history : list[dict[str, str]] | None
            過去のメッセージ履歴。[{"role": "user"/"assistant", "content": "..."}]
        model : str | None
            使用するモデル名。None の場合 self.model を使用。
        system : str | None
            システムプロンプト。
        temperature : float
            サンプリング温度。
        max_tokens : int
            最大トークン数。
        stream : bool
            True の場合ストリーミングジェネレータを返す。
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        resp = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
            stream=stream,
        )
        resp.raise_for_status()

        if stream:
            return self._stream_chat(resp)
        return resp.json()["message"]["content"]

    def _stream_chat(self, resp: requests.Response) -> Generator[str, None, None]:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                msg = data.get("message", {})
                yield msg.get("content", "")
                if data.get("done"):
                    break

    # ------------------------------------------------------------------
    # 埋め込み
    # ------------------------------------------------------------------

    def embed(
        self,
        text: str | list[str],
        model: str = "nomic-embed-text",
    ) -> list[float] | list[list[float]]:
        """テキストの埋め込みベクトルを取得する（/api/embed エンドポイント）。

        Parameters
        ----------
        text : str | list[str]
            埋め込むテキスト（単一または複数）。
        model : str
            埋め込みモデル名（デフォルト: nomic-embed-text）。

        Returns
        -------
        list[float] | list[list[float]]
            単一テキストなら 1D リスト、複数なら 2D リスト。
        """
        single = isinstance(text, str)
        inputs = [text] if single else text

        payload = {"model": model, "input": inputs}
        resp = self._session.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        return embeddings[0] if single else embeddings

    # ------------------------------------------------------------------
    # OpenAI互換エンドポイント
    # ------------------------------------------------------------------

    def openai_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """OpenAI互換エンドポイント (/v1/chat/completions) を使用する。

        openai Python ライブラリとの互換性が必要な場合に便利。
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def benchmark(
        self,
        prompt: str = "日本の首都はどこですか？",
        models: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """複数モデルの応答時間と内容を比較する。

        Parameters
        ----------
        prompt : str
            テスト用プロンプト。
        models : list[str] | None
            比較するモデル一覧。None の場合 list_models() から取得。

        Returns
        -------
        dict[str, dict[str, Any]]
            {モデル名: {"response": str, "elapsed_sec": float}} の辞書。
        """
        if models is None:
            models = self.list_models()

        results: dict[str, dict[str, Any]] = {}
        for m in models:
            t0 = time.time()
            try:
                response = self.generate(prompt, model=m, max_tokens=256)
                elapsed = time.time() - t0
                results[m] = {"response": response, "elapsed_sec": round(elapsed, 2), "error": None}
            except Exception as e:  # noqa: BLE001
                results[m] = {"response": None, "elapsed_sec": None, "error": str(e)}
        return results


# ------------------------------------------------------------------
# シンプルなファクトリ関数
# ------------------------------------------------------------------


def get_client(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> OllamaClient:
    """OllamaClientのインスタンスを取得するファクトリ関数。

    Examples
    --------
    >>> client = get_client("llama3.2:3b")
    >>> print(client.chat("Hello!"))
    """
    return OllamaClient(base_url=base_url, model=model)
