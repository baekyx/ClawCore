"""
工具4: 计算器工具
"""

from typing import Dict, Any, List, Optional
import math

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class CalculatorTool(Tool):
    """安全计算器 — 使用 Python math 模块，限制输入"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算。支持的运算：+, -, *, /, **, %, sqrt, sin, cos, log, abs",
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="expression", type="string",
                          description="数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'",
                          required=True),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        expr = parameters.get("expression", "")

        # 安全检查：只允许白名单字符
        allowed = set("0123456789+-*/().,% ^eE")
        for ch in expr:
            if ch not in allowed and ch not in "sqrtanlogbmcdfiph":
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"表达式包含不安全的字符: '{ch}'"
                )

        # 构建安全的执行环境
        safe_globals = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "abs": abs,
            "round": round,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
            "ceil": math.ceil,
            "floor": math.floor,
        }

        try:
            result = eval(expr, safe_globals, {})
            return ToolResponse.success(
                text=f"{expr} = {result}",
                data={"expression": expr, "result": result}
            )
        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"计算失败: {e}"
            )
