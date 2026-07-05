"""
Skill 版本管理器

语义版本 (SemVer): MAJOR.MINOR.PATCH
- MAJOR: 不兼容的变更（旧用法不再有效）
- MINOR: 新增功能，向后兼容
- PATCH: 修复/优化，完全向后兼容

变更历史追踪，支持回滚
"""

from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json


class SkillVersioning:
    """Skill 语义版本管理"""

    def __init__(self, versions_dir: str = "data/skill_artifacts/.versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def bump_version(self, current: str, change_type: str = "patch") -> str:
        """
        版本号升级

        change_type: "major" | "minor" | "patch"
        """
        parts = [int(x) for x in current.split(".")]
        if len(parts) != 3:
            parts = [0, 1, 0]

        if change_type == "major":
            return f"{parts[0]+1}.0.0"
        elif change_type == "minor":
            return f"{parts[0]}.{parts[1]+1}.0"
        else:  # patch
            return f"{parts[0]}.{parts[1]}.{parts[2]+1}"

    def save_version(self, skill_name: str, version: str,
                     content: str, changelog: str = "") -> Path:
        """保存一个历史版本"""
        skill_versions_dir = self.versions_dir / skill_name
        skill_versions_dir.mkdir(parents=True, exist_ok=True)

        version_path = skill_versions_dir / f"v{version}.md"
        version_path.write_text(content, encoding='utf-8')

        # 更新变更日志
        changelog_path = skill_versions_dir / "changelog.json"
        changelog_data = []
        if changelog_path.exists():
            changelog_data = json.loads(changelog_path.read_text(encoding='utf-8'))

        changelog_data.append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "changelog": changelog,
        })
        changelog_path.write_text(
            json.dumps(changelog_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        return version_path

    def rollback(self, skill_name: str, target_version: str) -> Optional[str]:
        """回滚到指定版本，返回回滚后的内容"""
        version_path = self.versions_dir / skill_name / f"v{target_version}.md"
        if not version_path.exists():
            return None
        return version_path.read_text(encoding='utf-8')

    def get_history(self, skill_name: str) -> List[Dict]:
        """获取技能的版本历史"""
        changelog_path = self.versions_dir / skill_name / "changelog.json"
        if not changelog_path.exists():
            return []
        return json.loads(changelog_path.read_text(encoding='utf-8'))

    def determine_change_type(self, old_body: str, new_body: str) -> str:
        """
        自动判断变更类型

        启发式：
        - 新增章节 "##" → minor
        - 纯文字修改 → patch
        - 参数变化 → major（需人工确认）
        """
        if old_body == new_body:
            return "patch"

        # 统计差异程度
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, old_body, new_body).ratio()

        # 统计新增行
        old_lines = set(old_body.strip().split('\n'))
        new_lines = set(new_body.strip().split('\n'))
        added = len(new_lines - old_lines)
        removed = len(old_lines - new_lines)

        if ratio < 0.5 or removed > 5:
            return "major"
        elif added > 3:
            return "minor"
        else:
            return "patch"
