"""测试纯逻辑工具：calculator, finish, task_manager"""
import pytest
from src.tools.calculator import CalculatorTool
from src.tools.finish import FinishTool
from src.tools.task_manager import TaskManagerTool


class TestCalculator:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_basic_arithmetic(self):
        r = self.tool.run({"expression": "2 + 3 * 4"})
        assert r.status.value == "success"
        assert "14" in r.text

    def test_sqrt(self):
        r = self.tool.run({"expression": "sqrt(16)"})
        assert r.status.value == "success"
        assert "4.0" in r.text

    def test_invalid_expression(self):
        r = self.tool.run({"expression": "import os"})
        assert r.status.value == "error"

    def test_safe_chars_blocked(self):
        """不安全字符应被拦截"""
        r = self.tool.run({"expression": "__import__('os')"})
        assert r.status.value == "error"

    def test_division_by_zero(self):
        r = self.tool.run({"expression": "1/0"})
        assert r.status.value == "error"


class TestFinish:
    def setup_method(self):
        self.tool = FinishTool()

    def test_finish_with_answer(self):
        r = self.tool.run({"answer": "答案是42"})
        assert r.status.value == "success"
        assert r.data["finished"] is True
        assert "42" in r.text

    def test_finish_with_summary(self):
        r = self.tool.run({"answer": "完成", "summary": "经过了3步推理"})
        assert "3步推理" in r.text


class TestTaskManager:
    def setup_method(self):
        self.tool = TaskManagerTool()

    def test_create_tasks(self):
        r = self.tool.run({
            "action": "create",
            "summary": "测试任务",
            "todos": [
                {"content": "任务1", "status": "completed"},
                {"content": "任务2", "status": "in_progress"},
                {"content": "任务3", "status": "pending"},
            ]
        })
        assert r.status.value == "success"
        assert r.data["total"] == 3
        assert r.data["completed"] == 1
        assert r.data["in_progress"] == 1

    def test_single_in_progress_enforced(self):
        """单线程强制：不许2个 in_progress"""
        r = self.tool.run({
            "action": "create",
            "todos": [
                {"content": "A", "status": "in_progress"},
                {"content": "B", "status": "in_progress"},
            ]
        })
        assert r.status.value == "error"

    def test_clear_tasks(self):
        self.tool.run({"action": "create", "todos": [
            {"content": "X", "status": "pending"}
        ]})
        r = self.tool.run({"action": "clear"})
        assert r.status.value == "success"
        s = self.tool.run({"action": "status"})
        assert s.data["total"] == 0

    def test_invalid_action(self):
        r = self.tool.run({"action": "unknown"})
        assert r.status.value == "success"  # defaults to status
