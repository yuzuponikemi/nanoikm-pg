"""
モデレーター（司会者）モジュール
議論のテーマ設定と最終合成を担当する
"""

from .debater import LLMClient

MODERATOR_SYSTEM = """あなたは公平な議論の司会者・ファシリテーターです。

役割:
- テーマを明確化し、議論の焦点を定める
- 全員の意見を公平に聞き、最後に統合する
- 特定の立場に肩入れせず、建設的な結論を導く
- 日本語で回答する"""


class Moderator:
    def __init__(self, client: LLMClient):
        self.client = client

    def open(self, topic: str) -> str:
        """議論の開幕宣言"""
        prompt = f"""テーマ「{topic}」の議論を始めます。

以下を行ってください（150字以内）：
1. このテーマの核心的な問いを1文で明確化する
2. 参加者に何を議論してほしいかを簡潔に伝える

「それでは〜」という書き出しで始めてください。"""
        return self.client.generate(MODERATOR_SYSTEM, prompt)

    def synthesize(self, topic: str, all_statements: list[dict]) -> str:
        """全発言を統合して最終結論を出す"""
        statements_text = "\n".join(
            [f"{s['emoji']} {s['name']} ({s['round_name']}): {s['text']}" for s in all_statements]
        )

        prompt = f"""テーマ「{topic}」について、以下の全発言を統合してください。

【全発言】
{statements_text}

【統合レポートの形式】（300字以内）
**主要な論点：** （2〜3点を箇条書き）
**合意できた点：** （あれば）
**意見が割れた点：** （あれば）
**推奨アクション：** 具体的な次のステップを1〜2文で

議論を踏まえた上で、中立的かつ実践的な結論を出してください。"""

        return self.client.generate(MODERATOR_SYSTEM, prompt)
