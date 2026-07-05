"""
Skill 生命周期管理器 —— 渐进式披露 + 双层沉淀

前向循环（使用期）：
- Level 1: 启动时加载 name+description (~100 tokens/skill)
- Level 2: Agent 通过 skill_invoke 工具按需加载完整 SKILL.md
- Level 3: 请求时加载 resources (scripts/examples/references)

后向循环（进化期）：
- 系统提示引导: Agent 实时识别可沉淀技能
- 后台 Review: 异步分析对话日志，提取新模式 → 生成候选 Skill → 验证 → 沉淀

复用 HelloAgents SkillLoader 的 SKILL.md 格式
"""

import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config.settings import SkillConfig


@dataclass
class SkillInfo:
    """技能元数据（Level 1）"""
    name: str
    description: str
    version: str = "0.1.0"
    tags: List[str] = field(default_factory=list)
    success_rate: float = 0.0     # 成功率 0.0-1.0
    usage_count: int = 0           # 使用次数
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at: self.created_at = now
        if not self.updated_at: self.updated_at = now


@dataclass
class SkillFull(SkillInfo):
    """完整技能信息（Level 2+3）"""
    body: str = ""                 # SKILL.md 正文
    dependencies: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class SkillManager:
    """
    Skill 生命周期管理器

    用法：
        manager = SkillManager(skills_dir="data/skill_artifacts", llm=llm)
        # 前向
        prompt = manager.get_skills_prompt(user_input, max_tokens=500)
        # 后向
        candidates = manager.run_review_cycle(session_logs)
    """

    def __init__(self, config: SkillConfig = None, llm=None):
        self.config = config or SkillConfig()
        self.llm = llm
        self.skills_dir = Path(self.config.skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # 启动时加载 Level 1 元数据
        self.skills: Dict[str, SkillFull] = {}   # 完整技能缓存
        self.metadata: Dict[str, SkillInfo] = {} # Level 1 元数据
        self._scan_skills()

    # === Level 1: 启动时扫描元数据 ===

    def _scan_skills(self):
        """扫描 skills_dir，只加载 YAML frontmatter（Level 1）"""
        if not self.skills_dir.exists():
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            frontmatter = self._parse_frontmatter(skill_md)
            if not frontmatter:
                continue

            name = frontmatter.get("name", skill_dir.name)
            self.metadata[name] = SkillInfo(
                name=name,
                description=frontmatter.get("description", ""),
                version=frontmatter.get("version", "0.1.0"),
                tags=frontmatter.get("tags", []),
                success_rate=frontmatter.get("success_rate", 0.0),
                usage_count=frontmatter.get("usage_count", 0),
            )

    def _parse_frontmatter(self, path: Path) -> Optional[Dict]:
        """解析 SKILL.md 的 YAML frontmatter"""
        try:
            content = path.read_text(encoding='utf-8')
        except Exception:
            return None

        match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if not match:
            return None

        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None

    # === Level 2: 按需加载完整技能 ===

    def get_skill(self, name: str) -> Optional[SkillFull]:
        """获取完整技能（加载 body）"""
        if name in self.skills:
            # 更新使用计数
            self.skills[name].usage_count += 1
            return self.skills[name]

        if name not in self.metadata:
            return None

        meta = self.metadata[name]
        skill_dir = self.skills_dir / name
        skill_md = skill_dir / "SKILL.md"

        try:
            content = skill_md.read_text(encoding='utf-8')
        except Exception:
            return None

        # 分离 frontmatter 和 body
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        if not match:
            return None

        _, body = match.groups()

        # 扫描 resources
        examples = self._list_dir_files(skill_dir / "examples")

        skill = SkillFull(
            name=meta.name,
            description=meta.description,
            version=meta.version,
            tags=meta.tags,
            success_rate=meta.success_rate,
            usage_count=meta.usage_count + 1,
            body=body.strip(),
            examples=examples,
        )
        self.skills[name] = skill
        return skill

    def _list_dir_files(self, dir_path: Path) -> List[str]:
        if not dir_path.exists():
            return []
        return [f.name for f in dir_path.iterdir() if f.is_file()][:5]

    # === 渐进式目录披露 ===

    def get_skills_prompt(self, user_input: str = "", max_tokens: int = 500) -> str:
        """
        生成可注入 System Prompt 的技能列表

        策略：关键词匹配筛选最相关的 N 个，控制 Token 预算
        """
        if not self.metadata:
            return ""

        # 关键词匹配评分
        scored = []
        query_lower = user_input.lower()
        for name, info in self.metadata.items():
            score = 0
            if name.lower() in query_lower:
                score += 3
            if any(tag.lower() in query_lower for tag in info.tags):
                score += 2
            # 名称相似加分
            desc_lower = info.description.lower()
            for word in query_lower.split():
                if word in desc_lower:
                    score += 1
            scored.append((score, info))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Token 预算控制 (~100 tokens/skill)
        lines = ["## 可用技能"]
        token_est = 0
        for score, info in scored[:self.config.max_skills_in_prompt]:
            line = f"- **{info.name}**: {info.description}"
            if info.usage_count > 10:
                line += f" (使用{info.usage_count}次, 成功率{info.success_rate:.0%})"
            token_est += len(line) // 4 + 10
            if token_est > self.config.max_tokens_per_skill_meta * len(lines):
                break
            lines.append(line)

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    # === 后向循环: Skill 自进化 ===

    def create_skill(self, name: str, description: str, body: str,
                     tags: List[str] = None, version: str = "0.1.0") -> Optional[Path]:
        """
        创建新 Skill — 写入 SKILL.md 到 skills_dir/{name}/
        """
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        frontmatter = {
            "name": name, "description": description,
            "version": version, "tags": tags or [],
            "created": datetime.now().strftime("%Y-%m-%d"),
            "success_rate": 0.0, "usage_count": 0,
        }

        content = "---\n" + yaml.dump(frontmatter, allow_unicode=True) + "---\n\n" + body

        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content, encoding='utf-8')

        # 更新内存
        self.metadata[name] = SkillInfo(
            name=name, description=description,
            version=version, tags=tags or [],
        )
        return skill_path

    def update_skill(self, name: str, body: str = None,
                     description: str = None, version: str = None) -> bool:
        """更新已有 Skill"""
        if name not in self.metadata:
            return False

        skill_dir = self.skills_dir / name
        skill_md = skill_dir / "SKILL.md"

        # 读取现有 content
        existing = skill_md.read_text(encoding='utf-8')
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', existing, re.DOTALL)
        if not match:
            return False

        fm_str, old_body = match.groups()
        frontmatter = yaml.safe_load(fm_str) or {}

        if description:
            frontmatter["description"] = description
        if version:
            frontmatter["version"] = version
        frontmatter["updated_at"] = datetime.now().isoformat()

        new_body = body if body is not None else old_body
        content = "---\n" + yaml.dump(frontmatter, allow_unicode=True) + "---\n\n" + new_body
        skill_md.write_text(content, encoding='utf-8')

        # 清缓存
        self.skills.pop(name, None)
        self.metadata[name] = SkillInfo(
            name=name,
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", "0.1.0"),
            tags=frontmatter.get("tags", []),
            success_rate=frontmatter.get("success_rate", 0.0),
            usage_count=frontmatter.get("usage_count", 0),
        )
        return True

    def record_usage(self, name: str, success: bool):
        """记录技能使用结果，更新成功率"""
        if name in self.metadata:
            info = self.metadata[name]
            old_successes = info.success_rate * info.usage_count
            info.usage_count += 1
            if success:
                info.success_rate = (old_successes + 1) / info.usage_count
            else:
                info.success_rate = old_successes / info.usage_count

    def list_skills(self) -> List[str]:
        return list(self.metadata.keys())

    def get_stats(self) -> Dict:
        return {
            "total_skills": len(self.metadata),
            "skills": [
                {"name": s.name, "usage": s.usage_count, "success_rate": s.success_rate}
                for s in sorted(self.metadata.values(), key=lambda x: x.usage_count, reverse=True)
            ]
        }
