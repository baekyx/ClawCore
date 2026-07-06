"""ClawCore Agent Loop — 基于 HelloAgents ReActAgent 的扩展版

核心扩展点（Phase 2-4 逐步接入）：
- 每轮自动检索相关记忆并注入上下文
- 上下文超预算时触发压缩流水线
- 自动匹配相关 Skill

Phase 1: 跑通基础 ReAct 循环 + 6 个工具
Phase 2: 多层记忆集成（会话记忆 + 工作记忆 + 长期记忆检索）
Phase 3: 上下文压缩四层流水线集成
Phase 4: Skill 自进化（渐进式披露 + 技能加载）
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

def _safe(s):
    """移除 Windows GBK 不兼容的字符"""
    return str(s).encode('gbk', errors='ignore').decode('gbk')

# 导入 HelloAgents
HELLO_AGENTS_PATH = Path("D:/code/HelloAgents")
if str(HELLO_AGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(HELLO_AGENTS_PATH))

from hello_agents.core.agent import Agent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.message import Message
from hello_agents.core.streaming import StreamEvent, StreamEventType
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.response import ToolStatus

from config.settings import ClawCoreConfig, get_config
from src.llm import create_llm
from src.context import ContextPipeline
from src.skills import SkillManager


# ClawCore 默认系统提示词
DEFAULT_SYSTEM_PROMPT = """你是一个有用的 AI 助手，具备以下能力：
- 使用工具完成用户的任务请求
- 在需要时搜索网络获取最新信息
- 读取和写入文件
- 逐步推理，先思考后行动

## 工作流程
1. 仔细分析用户的问题
2. 决定是否需要使用工具
3. 如果需要工具，选择合适的工具并调用
4. 观察工具返回结果，决定下一步
5. 当有足够信息时，给出最终答案

## 搜索策略（重要）
- web_search 返回的标题+摘要通常已足够回答问题，不要盲目抓取
- 只有当摘要信息不完整、需要原文细节时才调用 web_fetch
- 每次最多 fetch 1-2 个最有价值的链接，不要对每个结果都抓
- web_fetch 对动态页面(TheVerge/TechCrunch/SPA)无效，遇到 dynamic=true 直接跳过

## 记忆与技能
- 当用户让你"记住"某事，或对话中有重要信息时，使用 memory_write 工具保存
- 当需要回忆之前的信息时，使用 memory_recall 工具查询

## 注意事项
- 优先使用工具获取实时信息，而非依赖训练数据
- 文件操作前先确认路径正确
- 网络搜索时使用准确的关键词，新闻类带日期"""


class ClawCoreAgent(Agent):
    """
    ClawCore Agent — 包装 HelloAgents ReActAgent 核心循环

    Phase 1: 基础 ReAct 循环
    Phase 2: + 记忆注入
    Phase 3: + 上下文压缩
    Phase 4: + Skill 匹配
    """

    def __init__(
        self,
        name: str = "ClawCore",
        llm: Optional[HelloAgentsLLM] = None,
        system_prompt: Optional[str] = None,
        config: Optional[ClawCoreConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager = None,
        context_pipeline = None,
        skill_manager = None,
        max_steps: int = 10,
    ):
        self.clawcore_config = config or get_config()

        # 创建 LLM
        if llm is None:
            llm = create_llm(self.clawcore_config)

        # 创建工具注册表
        if tool_registry is None:
            tool_registry = ToolRegistry()

        # 转为 HelloAgents Config
        hconfig = self._build_helloagents_config()

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            config=hconfig,
            tool_registry=tool_registry,
        )

        self.max_steps = max_steps

        # 工具熔断: 连续失败计数 {tool_name: failure_count}
        self._tool_failures = {}
        self._tool_skip = set()  # 本轮已熔断的工具

        # Phase 2: 多层记忆管理器
        self.memory_manager = memory_manager

        # Phase 3: 上下文压缩流水线
        if context_pipeline is None and llm:
            context_pipeline = ContextPipeline(self.clawcore_config.agent, llm)
        self.context_pipeline = context_pipeline

        # Phase 4: Skill 管理器
        if skill_manager is None:
            skill_manager = SkillManager(self.clawcore_config.skill, llm)
        self.skill_manager = skill_manager

    def _build_helloagents_config(self):
        """将 ClawCoreConfig 转为 HelloAgents Config"""
        c = self.clawcore_config
        from hello_agents.core.config import Config as HConfig
        return HConfig(
            context_window=c.agent.context_window,
            compression_threshold=c.agent.compression_threshold,
            min_retain_rounds=c.agent.min_retain_rounds,
            trace_enabled=True,
            trace_dir="data/traces",
        )

    def run(self, input_text: str, **kwargs) -> str:
        """
        ClawCore Agent 主入口 — ReAct 循环

        扩展流程（Phase 逐步接入）：
        1. [Phase 2] 检索相关记忆
        2. [Phase 4] 匹配相关 Skill
        3. [Phase 3] 构建上下文（含压缩决策）
        4. 标准 ReAct 循环（继承 HelloAgents）
        """
        print(f"\n{'='*60}")
        print(f"[ClawCore] {self.name} 开始处理: {input_text[:80]}")
        print(f"{'='*60}")

        start_time = datetime.now()

        # Phase 2: 记忆检索
        memory_context = ""
        if self.memory_manager:
            memory_context = self.memory_manager.retrieve(input_text)
            if memory_context:
                print(f"[*] 检索到相关记忆 ({len(memory_context)} 字符)")

        # Phase 4: Skill 渐进式披露
        skills_prompt = ""
        if self.skill_manager and self.skill_manager.list_skills():
            skills_prompt = self.skill_manager.get_skills_prompt(input_text)
            if skills_prompt:
                skill_count = len(self.skill_manager.list_skills())
                print(f"[Skill] 可用技能: {skill_count} 个")

        # 构建消息列表
        messages = self._build_messages(input_text, memory_context, skills_prompt)

        # 构建工具 Schema
        tool_schemas = self._build_tool_schemas()

        current_step = 0
        final_answer = ""

        # === ReAct 主循环 ===
        while current_step < self.max_steps:
            current_step += 1
            step_start = datetime.now()
            print(f"\n--- 第 {current_step}/{self.max_steps} 步 ---")

            try:
                response = self.llm.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    **kwargs,
                )
            except Exception as e:
                final_answer = f"LLM 调用失败: {e}，请稍后重试"
                print(f"[ERR] {final_answer}")
                if self.trace_logger:
                    self.trace_logger.log_event("error", {
                        "error_type": type(e).__name__, "message": str(e)
                    }, step=current_step)
                break

            # 没有工具调用 → 返回答案
            tool_calls = response.tool_calls
            if not tool_calls:
                final_answer = response.content or "抱歉，我无法回答这个问题。"
                print(_safe(f"  {final_answer[:100]}..."))
                break

            # 将助手消息加入历史
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            # 执行所有工具调用
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_call_id = tool_call.id

                # 步骤超时 → 剩余工具发超时消息后 continue
                if (datetime.now() - step_start).total_seconds() > 30:
                    print(f"[!] 步骤超时, 标记跳过")
                    messages.append({"role": "tool", "tool_call_id": tool_call_id,
                        "content": "[ERR] 步骤超时(>30s), 工具未执行"})
                    continue

                try:
                    arguments = json.loads(tool_call.arguments)
                except json.JSONDecodeError as e:
                    err_msg = f"[ERR] 参数格式不正确: {e}"
                    messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": err_msg})
                    print(f"  {err_msg}")
                    continue

                # 工具熔断检查
                if tool_name in self._tool_skip:
                    messages.append({"role": "tool", "tool_call_id": tool_call_id,
                        "content": f"[ERR] {tool_name} 已熔断, 本轮不再调用"})
                    print(f"  [!] {tool_name} 已熔断, 跳过")
                    continue

                print(f">> {tool_name}({_safe(str(arguments))[:80]})")

                tool_start = datetime.now()
                try:
                    result = self._execute_tool_call(tool_name, arguments)
                    tool_elapsed = (datetime.now() - tool_start).total_seconds()
                    if tool_elapsed > 30:
                        print(f"  [!] {tool_name} 耗时 {tool_elapsed:.0f}s")
                except Exception as e:
                    result = f"[ERR] 工具执行异常: {e}"
                    if self.trace_logger:
                        self.trace_logger.log_event("tool_error", {
                            "tool_name": tool_name, "error": str(e)
                        }, step=current_step)

                # 更新熔断计数
                if result.startswith("[ERR]"):
                    self._tool_failures[tool_name] = self._tool_failures.get(tool_name, 0) + 1
                    if self._tool_failures[tool_name] >= 3:
                        self._tool_skip.add(tool_name)
                        print(f"  [!] {tool_name} 连续失败3次, 本轮熔断")
                    print(f"  {_safe(result[:120])}")
                else:
                    self._tool_failures[tool_name] = 0  # 成功重置
                    preview = _safe(result[:150]).replace('\n', ' ')
                    print(f"  => {preview}...")

                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

            # Phase 3: 上下文压缩决策
            if self.context_pipeline and self.context_pipeline.should_compress(messages):
                messages = self.context_pipeline.compress(messages)

        # 达到最大步数
        if current_step >= self.max_steps and not final_answer:
            final_answer = "抱歉，我无法在限定步数内完成这个任务。"

        # 保存对话历史
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        # Phase 2: 记录到会话记忆
        if self.memory_manager:
            self.memory_manager.remember_session_message("user", input_text)
            self.memory_manager.remember_session_message("assistant", final_answer)

        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"[OK] 完成 ({current_step} 步, {duration:.1f}秒)")
        print(f"{'='*60}")

        return final_answer

    def run_stream(self, input_text: str, **kwargs):
        """
        流式执行 Agent — 实时输出 LLM 文本块 + 工具调用事件

        Yields:
            StreamEvent: stream events (AGENT_START, LLM_CHUNK, TOOL_CALL_FINISH, AGENT_FINISH)
        """
        yield StreamEvent.create(StreamEventType.AGENT_START, self.name, input_text=input_text)

        # Phase 2: memory retrieval
        memory_context = ""
        if self.memory_manager:
            memory_context = self.memory_manager.retrieve(input_text)

        # Phase 4: skill matching
        skills_prompt = ""
        if self.skill_manager and self.skill_manager.list_skills():
            skills_prompt = self.skill_manager.get_skills_prompt(input_text)

        messages = self._build_messages(input_text, memory_context, skills_prompt)
        tool_schemas = self._build_tool_schemas()
        step = 0
        final_answer = ""

        while step < self.max_steps:
            step += 1
            step_start = datetime.now()
            yield StreamEvent.create(StreamEventType.STEP_START, self.name, step=step)

            # 先调 Function Calling 看有没有工具调用
            try:
                response = self.llm.invoke_with_tools(messages, tools=tool_schemas, tool_choice="auto", **kwargs)
            except Exception as e:
                final_answer = f"LLM 调用失败: {e}"
                yield StreamEvent.create(StreamEventType.ERROR, self.name, error=str(e))
                break

            tool_calls = response.tool_calls
            if not tool_calls:
                # 无工具调用 → 流式输出完整回答
                full_text = ""
                try:
                    for chunk in self.llm.stream_invoke(messages, **kwargs):
                        full_text += chunk
                        yield StreamEvent.create(StreamEventType.LLM_CHUNK, self.name, chunk=chunk)
                except Exception:
                    full_text = response.content or "无法回答"
                    yield StreamEvent.create(StreamEventType.LLM_CHUNK, self.name, chunk=full_text)
                final_answer = full_text
                break

            # 有工具调用 → 先输出 LLM 的思考文字，再执行工具
            if response.content:
                yield StreamEvent.create(StreamEventType.LLM_CHUNK, self.name,
                    chunk=response.content + "\n")

            messages.append({
                "role": "assistant", "content": response.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments}} for tc in tool_calls]
            })

            for tc in tool_calls:
                # 单步超时 → 给剩余tool_call发超时消息
                elapsed = (datetime.now() - step_start).total_seconds()
                if elapsed > 30:
                    yield StreamEvent.create(StreamEventType.LLM_CHUNK, self.name,
                        chunk=f"\n[!] 步骤超时({elapsed:.0f}s)\n")
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                        "content": f"[ERR] 步骤超时({elapsed:.0f}s), 工具未执行"})
                    continue

                try:
                    arguments = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                        "content": f"[ERR] 参数JSON格式错误"})
                    continue

                # 工具熔断检查
                if tc.name in self._tool_skip:
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                        "content": f"[ERR] {tc.name} 已熔断, 本轮不再调用"})
                    yield StreamEvent.create(StreamEventType.TOOL_CALL_FINISH, self.name,
                        tool_name=tc.name, result="已熔断, 跳过")
                    continue

                yield StreamEvent.create(StreamEventType.TOOL_CALL_START, self.name,
                    tool_name=tc.name, args=_safe(str(arguments)[:80]))

                try:
                    result = self._execute_tool_call(tc.name, arguments)
                except Exception as e:
                    result = f"[ERR] {e}"

                # 更新熔断计数
                if result.startswith("[ERR]"):
                    self._tool_failures[tc.name] = self._tool_failures.get(tc.name, 0) + 1
                    if self._tool_failures[tc.name] >= 3:
                        self._tool_skip.add(tc.name)

                yield StreamEvent.create(StreamEventType.TOOL_CALL_FINISH, self.name,
                    tool_name=tc.name, result=_safe(result[:200]))

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            # 压缩检查
            if self.context_pipeline and self.context_pipeline.should_compress(messages):
                messages = self.context_pipeline.compress(messages)

        if step >= self.max_steps and not final_answer:
            final_answer = "无法在限定步数内完成"

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        if self.memory_manager:
            self.memory_manager.remember_session_message("user", input_text)
            self.memory_manager.remember_session_message("assistant", final_answer)

        yield StreamEvent.create(StreamEventType.AGENT_FINISH, self.name, result=final_answer)

    def _build_messages(self, input_text: str, memory_context: str = "",
                        skills_prompt: str = "") -> List[Dict[str, str]]:
        """构建消息列表（支持 Phase 2 记忆 + Phase 4 Skill）"""
        messages = []

        # 系统提示词
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Phase 2: 注入记忆上下文
        if memory_context:
            messages.append({
                "role": "system",
                "content": f"## 你的记忆（来自过往对话的知识）\n{memory_context}"
            })

        # Phase 4: 注入可用 Skill 列表
        if skills_prompt:
            messages.append({
                "role": "system",
                "content": skills_prompt
            })

        # 历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 用户输入
        messages.append({"role": "user", "content": input_text})

        return messages

    def add_tool(self, tool):
        """添加工具"""
        self.tool_registry.register_tool(tool)

    def list_tools(self) -> list:
        """列出所有工具"""
        return self.tool_registry.list_tools() if self.tool_registry else []
