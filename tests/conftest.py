"""pytest 共享 fixtures"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 确保 HelloAgents 可导入
HELLO_AGENTS_PATH = Path("D:/code/HelloAgents")
if str(HELLO_AGENTS_PATH) not in sys.path:
    sys.path.insert(0, str(HELLO_AGENTS_PATH))
