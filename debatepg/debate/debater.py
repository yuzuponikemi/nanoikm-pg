"""
各ペルソナの発言生成モジュール
Ollama（デフォルト）または Anthropic Claude API を使用
"""

import json
import os
import requests
from typing import Optional

from .personas import Persona


class LLMClient:
    """LLMバックエンドの抽象クライアント"""

    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        ollama_host: str = "http://host.docker.internal:11434",
    ):
        self.backend = backend
        self.ollama_host = ollama_host

        if backend == "ollama":
            self.model = model or os.environ.get("DEBATE_MODEL", "llama3.2:latest")
        else:
            self.model = model or os.environ.get("DEBATE_MODEL", "claude-3-5-haiku-20241022")

    def generate(self, system_prompt: str, user_message: str) -> str:
        if self.backend == "ollama":
            return self._ollama(system_prompt, user_message)
        else:
            return self._anthropic(system_prompt, user_message)

    def _ollama(self, system_prompt: str, user_message: str) -> str:
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 512},
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Ollama に接続できません: {self.ollama_host}\n"
                "Mac で `ollama serve` が起動しているか確認してください。"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama エラー: {e}")

    def _anthropic(self, system_prompt: str, user_message: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic パッケージをインストールしてください: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 環境変数が設定されていません")

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return msg.content[0].text.strip()


class Debater:
    """単一ペルソナの発言生成"""

    def __init__(self, persona: Persona, client: LLMClient):
        self.persona = persona
        self.client = client

    def speak(
        self,
        topic: str,
        round_num: int,
        round_name: str,
        previous_statements: list[dict],
    ) -> str:
        """
        発言を生成する

        Args:
            topic: 議論テーマ
            round_num: ラウンド番号
            round_name: ラウンド名（"冒頭主張" など）
            previous_statements: これまでの発言リスト [{"name": ..., "text": ...}]
        """
        context = self._build_context(topic, round_num, round_name, previous_statements)
        return self.client.generate(self.persona.system_prompt, context)

    def _build_context(
        self,
        topic: str,
        round_num: int,
        round_name: str,
        previous_statements: list[dict],
    ) -> str:
        lines = [f"【議論テーマ】{topic}", f"【ラウンド {round_num}：{round_name}】", ""]

        if previous_statements:
            lines.append("【これまでの発言】")
            for s in previous_statements:
                lines.append(f"{s['emoji']} {s['name']}: {s['text']}")
            lines.append("")

        if round_num == 1:
            lines.append("このテーマについて、あなたの立場から冒頭主張を200字以内で述べてください。")
        elif round_num == 2:
            lines.append(
                "他の参加者の意見を踏まえ、あなたの立場から反論・補足を200字以内で述べてください。"
                "特に気になった意見に具体的に言及してください。"
            )
        elif round_num == 3:
            lines.append(
                "議論を踏まえ、最終的な立場を1〜2文で明確に述べてください。"
                "「結論として、〜」という形式で締めてください。"
            )
        else:
            lines.append(f"ラウンド {round_num} として、議論を深める発言を200字以内でしてください。")

        return "\n".join(lines)
