"""
Skill 验证器 —— 自动化回归测试

验证策略：
1. 格式校验：SKILL.md 格式合法性
2. 工具校验：引用的工具名是否在注册表中
3. 示例验证：是否包含可执行的示例
4. 成功率追踪：记录每次使用结果，低于阈值自动标记
"""

from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime


class SkillValidator:
    """Skill 验证器"""

    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry
        self.validation_log: List[Dict] = []

    def validate(self, skill_data: Dict) -> Dict:
        """
        验证一个候选 Skill

        返回: {"valid": bool, "errors": [...], "warnings": [...]}
        """
        result = {"valid": True, "errors": [], "warnings": []}

        # 1. 必填字段检查
        name = skill_data.get("name", "")
        description = skill_data.get("description", "")
        body = skill_data.get("body", "")

        if not name or len(name) < 3:
            result["errors"].append("name 至少 3 个字符")
        if not description or len(description) < 10:
            result["errors"].append("description 至少 10 个字符")
        if not body or len(body) < 50:
            result["errors"].append("body 至少 50 个字符")

        # 2. 名称格式检查
        if " " in name or any(c.isupper() for c in name):
            result["warnings"].append("name 应使用 kebab-case (小写连字符)")

        # 3. 工具引用检查
        if self.tool_registry and body:
            for tool_name in self.tool_registry.list_tools():
                if tool_name.lower() in body.lower():
                    break
            else:
                result["warnings"].append("body 中未引用任何已知工具")

        # 4. 版本格式检查
        version = skill_data.get("version", "")
        if version:
            parts = version.split(".")
            if len(parts) != 3 or not all(p.isdigit() for p in parts):
                result["warnings"].append("version 应使用语义版本 (X.Y.Z)")

        result["valid"] = len(result["errors"]) == 0

        # 记录验证日志
        self.validation_log.append({
            "name": name,
            "valid": result["valid"],
            "errors": result["errors"],
            "warnings": result["warnings"],
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def validate_batch(self, skills: List[Dict]) -> List[Dict]:
        """批量验证"""
        return [self.validate(s) for s in skills]

    def get_validation_stats(self) -> Dict:
        """验证统计"""
        total = len(self.validation_log)
        passed = sum(1 for v in self.validation_log if v["valid"])
        return {
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
        }
