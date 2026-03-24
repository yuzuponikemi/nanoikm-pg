#!/usr/bin/env python3
"""
debatepg — マルチエージェント議論シミュレーター
複数のAIペルソナが任意のテーマについて議論し、モデレーターが結論を出す

使用例:
  python main.py "RustをAIプロジェクトに採用すべきか"
  python main.py "LLMのファインチューニングは必要か" --save
  python main.py "Kubernetes導入の是非" --personas bull,bear,engineer
  python main.py "AIエージェントに意思決定を委ねるべきか" --backend claude
"""

import argparse
import sys

from rich.console import Console
from rich.table import Table

from debate.personas import PERSONAS, DEFAULT_PERSONAS
from debate.debater import LLMClient
from debate.session import DebateSession

console = Console()


def list_personas() -> None:
    table = Table(title="利用可能なペルソナ", show_header=True, header_style="bold cyan")
    table.add_column("キー", style="bold")
    table.add_column("名前")
    table.add_column("絵文字")
    table.add_column("立場", style="dim")
    for key, p in PERSONAS.items():
        table.add_row(key, p.name, p.emoji, p.stance)
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="マルチエージェント議論シミュレーター",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python main.py "RustをAIプロジェクトに採用すべきか"
  python main.py "LLMのファインチューニングは必要か" --save
  python main.py "Kubernetes導入の是非" --personas bull,bear,engineer
  python main.py "AIエージェントに意思決定を委ねるべきか" --backend claude --rounds 4
  python main.py --list-personas
        """,
    )

    parser.add_argument("topic", nargs="?", help="議論テーマ")
    parser.add_argument(
        "--backend",
        choices=["ollama", "claude"],
        default="ollama",
        help="LLMバックエンド (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="使用するモデル (ollama: llama3.2:latest, claude: claude-3-5-haiku-20241022)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://host.docker.internal:11434",
        help="Ollama ホスト URL (default: http://host.docker.internal:11434)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="議論ラウンド数 (default: 3)",
    )
    parser.add_argument(
        "--personas",
        default=",".join(DEFAULT_PERSONAS),
        help=f"参加ペルソナ（カンマ区切り, default: {','.join(DEFAULT_PERSONAS)}）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="議論結果をMarkdownファイルに保存する",
    )
    parser.add_argument(
        "--save-dir",
        default="transcripts",
        help="トランスクリプト保存ディレクトリ (default: transcripts/)",
    )
    parser.add_argument(
        "--list-personas",
        action="store_true",
        help="利用可能なペルソナ一覧を表示",
    )

    args = parser.parse_args()

    if args.list_personas:
        list_personas()
        return

    if not args.topic:
        parser.print_help()
        console.print("\n[yellow]⚠️  テーマを指定してください。例: python main.py \"RustをAIプロジェクトに採用すべきか\"[/yellow]")
        sys.exit(1)

    # ペルソナ検証
    persona_keys = [k.strip() for k in args.personas.split(",")]
    invalid = [k for k in persona_keys if k not in PERSONAS]
    if invalid:
        console.print(f"[red]❌ 不明なペルソナ: {', '.join(invalid)}[/red]")
        console.print(f"[dim]利用可能: {', '.join(PERSONAS.keys())}[/dim]")
        sys.exit(1)

    if len(persona_keys) < 2:
        console.print("[red]❌ ペルソナは2名以上必要です[/red]")
        sys.exit(1)

    # LLMクライアント初期化
    try:
        client = LLMClient(
            backend=args.backend,
            model=args.model,
            ollama_host=args.ollama_host,
        )
        console.print(
            f"[dim]🔧 バックエンド: {args.backend} | モデル: {client.model} | "
            f"ペルソナ: {len(persona_keys)}名 | {args.rounds}ラウンド[/dim]"
        )
    except Exception as e:
        console.print(f"[red]❌ 初期化エラー: {e}[/red]")
        sys.exit(1)

    # セッション実行
    session = DebateSession(
        topic=args.topic,
        persona_keys=persona_keys,
        client=client,
        num_rounds=args.rounds,
        save=args.save,
        save_dir=args.save_dir,
    )

    try:
        session.run()
    except ConnectionError as e:
        console.print(f"\n[red]❌ 接続エラー: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  中断されました[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ エラー: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
