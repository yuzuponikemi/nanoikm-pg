"""
ディベートセッション管理
ラウンド制御・発言収集・トランスクリプト出力
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from .personas import Persona, PERSONAS
from .debater import Debater, LLMClient
from .moderator import Moderator

console = Console()

ROUND_NAMES = {
    1: "冒頭主張",
    2: "反論・補足",
    3: "最終立場",
}


class DebateSession:
    def __init__(
        self,
        topic: str,
        persona_keys: list[str],
        client: LLMClient,
        num_rounds: int = 3,
        save: bool = False,
        save_dir: str = "transcripts",
    ):
        self.topic = topic
        self.num_rounds = num_rounds
        self.save = save
        self.save_dir = Path(save_dir)

        # ペルソナ・エージェント初期化
        self.personas: list[Persona] = [PERSONAS[k] for k in persona_keys]
        self.debaters: list[Debater] = [Debater(p, client) for p in self.personas]
        self.moderator = Moderator(client)

        # 発言履歴
        self.all_statements: list[dict] = []

    def run(self) -> None:
        """セッション全体を実行"""
        console.print()
        console.print(
            Panel(
                f"[bold white]{self.topic}[/bold white]",
                title="[bold cyan]🎙️  マルチエージェント議論シミュレーター[/bold cyan]",
                subtitle=f"[dim]参加者: {' '.join(p.emoji + p.name for p in self.personas)} | {self.num_rounds}ラウンド[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        console.print()

        # モデレーター開幕
        console.print(Rule("[bold cyan]📢 開幕[/bold cyan]", style="cyan"))
        opening = self._generate_with_spinner("モデレーター", "⚖️", "cyan", lambda: self.moderator.open(self.topic))
        console.print(
            Panel(opening, title="[cyan]⚖️ Moderator[/cyan]", border_style="cyan", padding=(0, 1))
        )
        console.print()

        # ラウンド実行
        for round_num in range(1, self.num_rounds + 1):
            round_name = ROUND_NAMES.get(round_num, f"ラウンド {round_num}")
            console.print(Rule(f"[bold white]Round {round_num}：{round_name}[/bold white]"))
            console.print()

            for debater in self.debaters:
                statement = self._generate_with_spinner(
                    debater.persona.name,
                    debater.persona.emoji,
                    debater.persona.color,
                    lambda d=debater: d.speak(
                        self.topic, round_num, round_name, self.all_statements
                    ),
                )
                entry = {
                    "round": round_num,
                    "round_name": round_name,
                    "name": debater.persona.name,
                    "emoji": debater.persona.emoji,
                    "stance": debater.persona.stance,
                    "text": statement,
                }
                self.all_statements.append(entry)

                console.print(
                    Panel(
                        statement,
                        title=f"[{debater.persona.color}]{debater.persona.emoji} {debater.persona.name}[/{debater.persona.color}] [dim]— {debater.persona.stance}[/dim]",
                        border_style=debater.persona.color.replace("bold ", ""),
                        padding=(0, 1),
                    )
                )
                console.print()

        # モデレーター最終合成
        console.print(Rule("[bold cyan]📋 最終合成[/bold cyan]", style="cyan"))
        synthesis = self._generate_with_spinner(
            "モデレーター（統合）", "⚖️", "cyan", lambda: self.moderator.synthesize(self.topic, self.all_statements)
        )
        console.print(
            Panel(
                synthesis,
                title="[bold cyan]⚖️ Moderator — 総括[/bold cyan]",
                border_style="cyan",
                padding=(1, 1),
            )
        )
        console.print()

        # トランスクリプト保存
        if self.save:
            self._save_transcript(synthesis)

    def _generate_with_spinner(self, name: str, emoji: str, color: str, fn) -> str:
        """スピナーを表示しながら生成"""
        with Live(
            Text(f"  {emoji} {name} が考えています...", style=f"dim {color.replace('bold ', '')}"),
            console=console,
            transient=True,
            refresh_per_second=10,
        ):
            result = fn()
        return result

    def _save_transcript(self, synthesis: str) -> None:
        """トランスクリプトをMarkdownファイルに保存"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = self.topic[:30].replace(" ", "_").replace("/", "-")
        filepath = self.save_dir / f"{timestamp}_{safe_topic}.md"

        lines = [
            f"# 議論トランスクリプト",
            f"",
            f"**テーマ**: {self.topic}",
            f"**日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**参加者**: {', '.join(p.emoji + p.name for p in self.personas)}",
            f"",
        ]

        current_round = 0
        for s in self.all_statements:
            if s["round"] != current_round:
                current_round = s["round"]
                lines.append(f"## Round {current_round}：{s['round_name']}")
                lines.append("")
            lines.append(f"### {s['emoji']} {s['name']} — {s['stance']}")
            lines.append(s["text"])
            lines.append("")

        lines += [
            "## 最終合成",
            "",
            synthesis,
            "",
        ]

        filepath.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[dim]💾 トランスクリプト保存: {filepath}[/dim]")
