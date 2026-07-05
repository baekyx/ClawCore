"""MyClaw CLI — 命令行交互入口"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import typer
from rich.console import Console
from rich.markdown import Markdown

from src.agent_loop.react_loop import MyClawAgent
from src.tools import create_default_registry
from src.llm import create_llm
from src.memory import MemoryManager
from src.skills import SkillManager
from config.settings import get_config

app = typer.Typer(
    name="myclaw",
    help="MyClaw — 基于多层记忆与 Skill 自进化的通用 Agent 助手",
)
console = Console()


@app.command()
def chat(
    query: str = typer.Option(None, "--query", "-q", help="单次问答"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="交互式多轮对话"),
    model: str = typer.Option(None, "--model", "-m", help="指定模型"),
):
    """
    MyClaw 智能对话

    示例:
        myclaw -q "今天天气怎么样"
        myclaw -i
    """
    config = get_config()
    if model:
        config.llm.model = model

    console.print(f"[bold cyan]🤖 MyClaw v0.1.0[/bold cyan]")
    console.print(f"[dim]模型: {config.llm.model}[/dim]")

    # 创建记忆管理器
    console.print(f"[dim]模型: {config.llm.model}[/dim]")
    memory = MemoryManager(config)
    console.print(f"[dim]记忆: 会话({memory.session.current_session_id or '新会话'}) | 长期({memory.long_term.count() if memory.long_term else 0}条)[/dim]")

    # 创建 Skill 管理器
    skill = SkillManager(config.skill, llm)

    # 创建 Agent
    llm = create_llm(config)
    registry = create_default_registry(skill_manager=skill)
    agent = MyClawAgent(
        name="MyClaw",
        llm=llm,
        config=config,
        tool_registry=registry,
        memory_manager=memory,
        skill_manager=skill,
        max_steps=config.agent.max_steps,
    )

    console.print(f"[dim]工具: {', '.join(agent.list_tools())[:80]}...[/dim]")

    if interactive:
        _run_interactive(agent)
    elif query:
        _run_single(agent, query)
    else:
        console.print("[yellow]请使用 -q 提问或 -i 进入交互模式[/yellow]")
        console.print("示例: myclaw -q '你好'  或  myclaw -i")


def _run_single(agent: MyClawAgent, query: str):
    """单次问答"""
    try:
        result = agent.run(query)
        console.print()
        console.print("[bold green]📝 回答:[/bold green]")
        console.print(Markdown(result))
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ 用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ 错误: {e}[/red]")


def _run_interactive(agent: MyClawAgent):
    """交互式多轮对话"""
    console.print("\n[bold]输入 '/exit' 退出, '/clear' 清空历史[/bold]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]你: [/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not user_input.strip():
            continue

        if user_input.lower() == "/exit":
            console.print("[yellow]再见！[/yellow]")
            break

        if user_input.lower() == "/clear":
            agent.clear_history()
            console.print("[dim]历史已清空[/dim]")
            continue

        try:
            result = agent.run(user_input)
            console.print()
            console.print("[bold green]MyClaw: [/bold green]")
            console.print(Markdown(result))
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ 中断，输入新问题继续[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ 错误: {e}[/red]")


if __name__ == "__main__":
    app()
