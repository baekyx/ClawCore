"""测试 L1 会话态记忆 (SQLite)"""
import pytest
import tempfile
import os
from src.memory.session_memory import SessionMemory, SessionMessage


class TestSessionMemory:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.mem = SessionMemory(db_path=self.tmp.name)

    def teardown_method(self):
        if hasattr(self.mem, '_conn') and self.mem._conn:
            self.mem._conn.close()
        import gc
        gc.collect()
        try:
            os.unlink(self.tmp.name)
        except PermissionError:
            pass  # Windows 可能锁文件

    def test_create_session(self):
        sid = self.mem.create_session("测试会话")
        assert sid.startswith("s-")
        assert self.mem.current_session_id == sid

    def test_add_and_get_messages(self):
        self.mem.create_session()
        self.mem.add_message(SessionMessage(role="user", content="你好"))
        self.mem.add_message(SessionMessage(role="assistant", content="你好！有什么可以帮你？"))

        msgs = self.mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_get_recent_for_context(self):
        self.mem.create_session()
        for i in range(10):
            self.mem.add_message(SessionMessage(role="user", content=f"消息{i}" * 100))
            self.mem.add_message(SessionMessage(role="assistant", content=f"回复{i}" * 100))

        recent = self.mem.get_recent_for_context(max_tokens=500)
        # 应该只返回最近的几条（500 tokens * 4 = 2000 字符以内）
        assert len(recent) < 10

    def test_list_sessions(self):
        import time
        sid1 = self.mem.create_session("会话1")
        time.sleep(0.01)  # 避免 ID 碰撞
        sid2 = self.mem.create_session("会话2")
        sessions = self.mem.list_sessions()
        assert len(sessions) == 2
        assert sid1 != sid2

    def test_delete_session(self):
        sid = self.mem.create_session("待删除")
        self.mem.add_message(SessionMessage(role="user", content="test"))
        self.mem.delete_session(sid)
        assert len(self.mem.list_sessions()) == 0

    def test_empty_session_messages(self):
        self.mem.create_session()
        assert self.mem.get_messages() == []
