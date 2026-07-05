"""
工具: Skill 调用工具

允许 Agent 按需加载完整技能内容。
渐进式披露 Level 2: 从元数据 → 完整 body
"""

from typing import Dict, Any, List

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class SkillInvokeTool(Tool):
    """Skill 调用工具 — Agent 按需加载技能"""

    def __init__(self, skill_manager=None):
        self.skill_manager = skill_manager

        # 动态生成描述（包含可用技能列表）
        if skill_manager:
            skills_list = skill_manager.list_skills()
            desc = "按需加载和调用技能。\n\n可用技能：\n"
            desc += skill_manager.get_skills_prompt(max_tokens=1000)
            if not skills_list:
                desc += "（暂无技能，可以通过使用中沉淀新技能）"
        else:
            desc = "按需加载和调用技能"

        super().__init__(
            name="skill_invoke",
            description=desc,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="skill", type="string",
                          description="要加载的技能名称", required=True),
            ToolParameter(name="args", type="string",
                          description="可选参数，替换技能中的 $ARGUMENTS 占位符",
                          required=False, default=""),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        skill_name = parameters.get("skill", "")
        args = parameters.get("args", "")

        if not self.skill_manager:
            return ToolResponse.error(
                code=ToolErrorCode.INTERNAL_ERROR,
                message="Skill 管理器未初始化"
            )

        skill = self.skill_manager.get_skill(skill_name)
        if not skill:
            available = ", ".join(self.skill_manager.list_skills())
            return ToolResponse.error(
                code=ToolErrorCode.NOT_FOUND,
                message=f"技能 '{skill_name}' 不存在。可用: {available}"
            )

        # 替换参数占位符
        body = skill.body.replace("$ARGUMENTS", args)

        # 构建完整技能内容
        content = f"""<skill name="{skill.name}" version="{skill.version}">
{body}
</skill>

✅ 技能已加载: {skill.name}
📝 {skill.description}

请严格遵循上述技能说明来完成任务。"""

        return ToolResponse.success(
            text=content,
            data={
                "name": skill.name,
                "version": skill.version,
                "loaded": True,
                "has_examples": len(skill.examples) > 0,
            }
        )
