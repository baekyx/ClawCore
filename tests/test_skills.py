"""
验证 Skill 自进化系统: 管理器/提取器/验证器/版本器
"""
import sys
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.skills import SkillManager, SkillExtractor, SkillValidator, SkillVersioning


# ── SkillManager: 创建/更新/列取/使用追踪 ──

class TestSkillManager:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        from config.settings import SkillConfig
        self.mgr = SkillManager(SkillConfig(skills_dir=self.tmp))

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_skill(self):
        path = self.mgr.create_skill(
            name="test-calc",
            description="测试计算器技能",
            body="## 步骤\n1. 使用 calculator\n2. 返回结果",
            tags=["math", "test"],
            version="0.1.0"
        )
        assert path is not None
        assert path.exists()
        assert "test-calc" in self.mgr.list_skills()

    def test_get_skill_loads_body(self):
        self.mgr.create_skill("reader", "读取文件", "## 步骤\n1. file_read\n2. 分析内容")
        skill = self.mgr.get_skill("reader")
        assert skill is not None
        assert "file_read" in skill.body
        assert skill.usage_count == 1  # get_skill 自动计次

    def test_update_skill(self):
        self.mgr.create_skill("updater", "原始描述", "原始内容")
        ok = self.mgr.update_skill("updater", body="新内容", description="新描述")
        assert ok
        skill = self.mgr.get_skill("updater")
        assert "新内容" in skill.body

    def test_record_usage_tracks_success_rate(self):
        self.mgr.create_skill("tracker", "追踪", "body")
        self.mgr.record_usage("tracker", success=True)
        self.mgr.record_usage("tracker", success=True)
        self.mgr.record_usage("tracker", success=False)
        assert self.mgr.metadata["tracker"].success_rate == 2 / 3

    def test_get_skills_prompt(self):
        self.mgr.create_skill("math-solver", "数学求解", "body", tags=["math"])
        self.mgr.create_skill("file-reader", "文件读取", "body", tags=["file"])
        prompt = self.mgr.get_skills_prompt("帮我算一道数学题", max_tokens=500)
        assert "math-solver" in prompt
        assert "可用技能" in prompt

    def test_nonexistent_skill_returns_none(self):
        assert self.mgr.get_skill("不存在") is None

    def test_list_skills_empty(self):
        assert self.mgr.list_skills() == []

    def test_stats(self):
        self.mgr.create_skill("s1", "d1", "body")
        self.mgr.create_skill("s2", "d2", "body")
        stats = self.mgr.get_stats()
        assert stats["total_skills"] == 2


# ── SkillExtractor: 模式挖掘 ──

class TestSkillExtractor:
    def setup_method(self):
        self.extractor = SkillExtractor(llm=None)

    def test_extract_tool_sequences(self):
        logs = [{
            "session_id": "s1",
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"name": "web_search"}, {"name": "web_fetch"}]},
                {"role": "assistant", "tool_calls": [
                    {"name": "file_write"}]},
            ]
        }]
        seqs = self.extractor._extract_tool_sequences(logs)
        assert len(seqs) == 1
        assert seqs[0]["sequence"] == ["web_search", "web_fetch", "file_write"]

    def test_find_frequent_sequences(self):
        seqs = [
            {"session_id": "s1", "sequence": ["web_search", "web_fetch", "file_write"]},
            {"session_id": "s2", "sequence": ["web_search", "web_fetch", "file_write"]},
            {"session_id": "s3", "sequence": ["web_search", "web_fetch", "file_write"]},
        ]
        freq = self.extractor._find_frequent_sequences(seqs, min_support=3)
        assert ("web_search", "web_fetch", "file_write") in freq

    def test_no_pattern_below_min_support(self):
        seqs = [
            {"session_id": "s1", "sequence": ["a", "b"]},
            {"session_id": "s2", "sequence": ["a", "b"]},
        ]
        freq = self.extractor._find_frequent_sequences(seqs, min_support=3)
        assert ("a", "b") not in freq

    def test_subsequence_detection(self):
        seqs = [
            {"session_id": "s1", "sequence": ["x", "y", "z", "x", "y"]},
            {"session_id": "s2", "sequence": ["x", "y", "z", "x", "y"]},
            {"session_id": "s3", "sequence": ["x", "y", "z", "x", "y"]},
        ]
        freq = self.extractor._find_frequent_sequences(seqs, min_support=3)
        # 连续子序列 "x","y","z" 出现3次
        assert ("x", "y", "z") in freq

    def test_extract_patterns_no_llm(self):
        """无 LLM 时仍能提取模式（描述用默认文本）"""
        logs = [
            {
                "session_id": "s1",
                "messages": [
                    {"role": "user", "content": "搜索AI新闻"},
                    {"role": "assistant", "tool_calls": [
                        {"name": "web_search"}, {"name": "web_fetch"}]},
                    {"role": "assistant", "content": "结果如下..."},
                ]
            },
            {
                "session_id": "s2",
                "messages": [
                    {"role": "user", "content": "搜索科技新闻"},
                    {"role": "assistant", "tool_calls": [
                        {"name": "web_search"}, {"name": "web_fetch"}]},
                ]
            },
            {
                "session_id": "s3",
                "messages": [
                    {"role": "user", "content": "搜索财经新闻"},
                    {"role": "assistant", "tool_calls": [
                        {"name": "web_search"}, {"name": "web_fetch"}]},
                ]
            },
        ]
        patterns = self.extractor.extract_patterns(logs)
        assert len(patterns) >= 1
        p = patterns[0]
        assert p.frequency >= 3
        assert "web_search" in p.tool_sequence


# ── SkillValidator: 格式/工具引用/版本校验 ──

class TestSkillValidator:
    def setup_method(self):
        self.v = SkillValidator()

    def test_valid_skill(self):
        r = self.v.validate({
            "name": "test-skill",
            "description": "这是一个测试技能，用于验证",
            "body": "## 步骤\n1. 使用 file_read 读取\n2. 分析内容并返回结果\n3. 如果需要，调用 calculator",
            "version": "1.0.0",
        })
        assert r["valid"]

    def test_missing_name(self):
        r = self.v.validate({"name": "ab", "description": "x" * 20, "body": "x" * 60})
        assert not r["valid"]

    def test_short_description(self):
        r = self.v.validate({"name": "test", "description": "短", "body": "x" * 60})
        assert not r["valid"]

    def test_kebab_case_warning(self):
        r = self.v.validate({"name": "Test Skill", "description": "x" * 20, "body": "x" * 60})
        assert len(r["warnings"]) >= 1

    def test_version_format_warning(self):
        r = self.v.validate({
            "name": "test", "description": "x" * 20, "body": "x" * 60,
            "version": "v1"
        })
        assert any("版本" in w for w in r["warnings"])

    def test_batch_validation(self):
        skills = [
            {"name": "good", "description": "x" * 20, "body": "x" * 60},
            {"name": "bad", "description": "x", "body": "x"},
        ]
        results = self.v.validate_batch(skills)
        assert results[0]["valid"]
        assert not results[1]["valid"]


# ── SkillVersioning: 语义版本/历史/回滚 ──

class TestSkillVersioning:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.sv = SkillVersioning(versions_dir=f"{self.tmp}/.versions")

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_bump_patch(self):
        assert self.sv.bump_version("1.0.0", "patch") == "1.0.1"

    def test_bump_minor(self):
        assert self.sv.bump_version("1.0.5", "minor") == "1.1.0"

    def test_bump_major(self):
        assert self.sv.bump_version("2.3.1", "major") == "3.0.0"

    def test_save_and_rollback(self):
        self.sv.save_version("test-skill", "1.0.0", "原始内容", "初始版本")
        self.sv.save_version("test-skill", "1.1.0", "新增了步骤3", "新增功能")

        rolled = self.sv.rollback("test-skill", "1.0.0")
        assert rolled == "原始内容"

        history = self.sv.get_history("test-skill")
        assert len(history) == 2

    def test_determine_change_type(self):
        # 小改 → patch (内容相同)
        assert self.sv.determine_change_type("hello world", "hello world") == "patch"
        # 新增少量行 → minor
        base = "步骤1: 读取文件\n步骤2: 分析内容\n步骤3: 输出结果"
        new = base + "\n步骤4: 格式化\n步骤5: 保存\n步骤6: 通知\n步骤7: 归档"
        assert self.sv.determine_change_type(base, new) == "minor"
        # 大幅改动 → major
        assert self.sv.determine_change_type("abcdefghij", "klmnopqrst") == "major"
