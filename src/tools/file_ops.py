"""
工具1-3: 文件操作工具
复用 HelloAgents file_tools.py 的核心逻辑，支持乐观锁冲突检测
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.errors import ToolErrorCode


class FileReadTool(Tool):
    """文件读取工具 — 支持 offset/limit，自动缓存 mtime"""

    def __init__(self, working_dir: str = "."):
        super().__init__(
            name="file_read",
            description="读取文件内容或列出目录内容",
        )
        self.working_dir = Path(working_dir).resolve()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string",
                          description="文件或目录路径", required=True),
            ToolParameter(name="offset", type="integer",
                          description="起始行号（0开始）", required=False, default=0),
            ToolParameter(name="limit", type="integer",
                          description="最大行数", required=False, default=2000),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path", "")
        offset = parameters.get("offset", 0)
        limit = parameters.get("limit", 2000)

        full_path = self._resolve(path)
        if not full_path.exists():
            return ToolResponse.error(code=ToolErrorCode.NOT_FOUND,
                                       message=f"路径不存在: {path}")

        if full_path.is_dir():
            return self._list_dir(path, full_path)

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            total = len(lines)
            lines = lines[offset:][:limit]

            return ToolResponse.success(
                text=f"读取 {len(lines)}/{total} 行",
                data={"content": ''.join(lines), "total_lines": total,
                      "file_mtime_ms": int(os.path.getmtime(full_path) * 1000)}
            )
        except Exception as e:
            return ToolResponse.error(code=ToolErrorCode.INTERNAL_ERROR,
                                       message=str(e))

    def _list_dir(self, path: str, full_path: Path) -> ToolResponse:
        entries = []
        for entry in sorted(full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                entries.append({
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if not entry.is_dir() else None,
                })
            except Exception:
                continue
        return ToolResponse.success(
            text=f"目录 {path}: {len(entries)} 个条目",
            data={"path": path, "entries": entries}
        )

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.working_dir / p


class FileWriteTool(Tool):
    """文件写入工具 — 原子写入 + 自动备份"""

    def __init__(self, working_dir: str = "."):
        super().__init__(
            name="file_write",
            description="创建或覆盖文件，自动备份原文件",
        )
        self.working_dir = Path(working_dir).resolve()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string",
                          description="文件路径", required=True),
            ToolParameter(name="content", type="string",
                          description="文件内容", required=True),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path", "")
        content = parameters.get("content", "")

        full_path = self._resolve(path)

        try:
            # 备份原文件（如果存在）
            if full_path.exists():
                backup_dir = full_path.parent / ".backups"
                backup_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(full_path, backup_dir / f"{full_path.name}.{ts}.bak")
            else:
                full_path.parent.mkdir(parents=True, exist_ok=True)

            # 原子写入
            tmp = full_path.with_suffix(full_path.suffix + '.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(tmp, full_path)

            return ToolResponse.success(
                text=f"写入 {path} ({len(content.encode('utf-8'))} 字节)",
                data={"written": True, "size_bytes": len(content.encode('utf-8'))}
            )
        except Exception as e:
            return ToolResponse.error(code=ToolErrorCode.INTERNAL_ERROR,
                                       message=str(e))

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.working_dir / p


class FileEditTool(Tool):
    """文件编辑工具 — 精确替换（old_string 唯一匹配）"""

    def __init__(self, working_dir: str = "."):
        super().__init__(
            name="file_edit",
            description="精确替换文件内容，old_string 必须唯一匹配",
        )
        self.working_dir = Path(working_dir).resolve()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="path", type="string",
                          description="文件路径", required=True),
            ToolParameter(name="old_string", type="string",
                          description="要替换的内容（必须唯一匹配）", required=True),
            ToolParameter(name="new_string", type="string",
                          description="替换后的内容", required=True),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        path = parameters.get("path", "")
        old = parameters.get("old_string", "")
        new = parameters.get("new_string", "")

        full_path = self._resolve(path)
        if not full_path.exists():
            return ToolResponse.error(code=ToolErrorCode.NOT_FOUND,
                                       message=f"文件不存在: {path}")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            matches = content.count(old)
            if matches != 1:
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message=f"old_string 必须唯一匹配，找到 {matches} 处"
                )

            # 备份
            backup_dir = full_path.parent / ".backups"
            backup_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(full_path, backup_dir / f"{full_path.name}.{ts}.bak")

            # 执行替换
            new_content = content.replace(old, new)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            delta = len(new.encode('utf-8')) - len(old.encode('utf-8'))
            return ToolResponse.success(
                text=f"编辑 {path} (变化 {delta:+d} 字节)",
                data={"modified": True, "changed_bytes": delta}
            )
        except Exception as e:
            return ToolResponse.error(code=ToolErrorCode.INTERNAL_ERROR,
                                       message=str(e))

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.working_dir / p
