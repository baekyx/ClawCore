"""ClawCore CLI — 命令行交互入口"""

import sys
from pathlib import Path

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

app = typer.Typer(name="clawcore", help="ClawCore — 多层记忆 + Skill 自进化 Agent")
console = Console()


@app.command()
def chat(
    query: str = typer.Option(None, "--query", "-q", help="单次问答"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="交互式多轮对话"),
    model: str = typer.Option(None, "--model", "-m", help="指定模型"),
):
    """ClawCore 智能对话"""
    config = get_config()
    if model:
        config.llm.model = model

    console.print(f"[bold cyan]ClawCore v0.1.0[/bold cyan]")
    console.print(f"[dim]模型: {config.llm.model}[/dim]")

    # 创建 LLM（Skill 需要，先创建）
    llm = create_llm(config)

    # 创建记忆管理器（Postgres 不可用时降级）
    memory = MemoryManager(config)
    mem_info = f"会话({memory.session.current_session_id or '新会话'})"
    if memory.long_term:
        mem_info += f" | 长期({memory.long_term.count()}条)"
    console.print(f"[dim]记忆: {mem_info}[/dim]")

    # 创建 Skill 管理器
    skill = SkillManager(config.skill, llm)
    console.print(f"[dim]技能: {len(skill.list_skills())} 个[/dim]")

    # 创建 Agent
    registry = create_default_registry(skill_manager=skill)
    agent = MyClawAgent(
        name="ClawCore",
        llm=llm,
        config=config,
        tool_registry=registry,
        memory_manager=memory,
        skill_manager=skill,
        max_steps=config.agent.max_steps,
    )

    console.print(f"[dim]工具: {', '.join(agent.list_tools())}[/dim]")

    if interactive:
        _run_interactive(agent)
    elif query:
        _run_single(agent, query)
    else:
        console.print("[yellow]请使用 -q 提问或 -i 进入交互模式[/yellow]")


def _run_single(agent: MyClawAgent, query: str):
    try:
        result = agent.run(query)
        console.print()
        console.print("[bold green]=== 回答 ===[/bold green]")
        console.print(Markdown(result))
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")


def _run_interactive(agent: MyClawAgent):
    console.print("\n[bold]输入 /exit 退出, /clear 清空历史[/bold]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You: [/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见[/yellow]")
            break

        if not user_input.strip():
            continue
        if user_input.lower() == "/exit":
            console.print("[yellow]再见[/yellow]")
            break
        if user_input.lower() == "/clear":
            agent.clear_history()
            console.print("[dim]历史已清空[/dim]")
            continue

        try:
            result = agent.run(user_input)
            console.print()
            console.print("[bold green]ClawCore:[/bold green]")
            console.print(Markdown(result))
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]中断，输入新问题继续[/yellow]")
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")


if __name__ == "__main__":
    app()
