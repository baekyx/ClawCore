# ClawCore

> 多层记忆 + Skill 自进化的通用 Agent 助手 | 3341 行 Python | 36 模块

ClawCore 是一个从零构建的 AI Agent 框架。三层记忆突破上下文窗口限制，Skill 双层沉淀让 Agent 越用越聪明。

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/baekyx/ClawCore.git
cd ClawCore

# 2. 配置
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY

# 3. 启动 Postgres（可选，不启动则降级运行）
docker run -d --name clawcore-pg \
  -e POSTGRES_PASSWORD=clawcore123 \
  -e POSTGRES_DB=clawcore \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行
python src/cli.py -q "你好"
python src/cli.py -i          # 交互模式
```

## 架构

```
用户输入 → CLI (Typer)
             │
      ClawCoreAgent (ReAct 循环)
      ┌────┼────┬────┬────┐
      ▼    ▼    ▼    ▼    ▼
   9工具  记忆  压缩  Skill  LLM
```

### 四大模块

| 模块 | 文件 | 核心功能 |
|------|------|---------|
| **Agent Loop** | `src/agent_loop/` | ReAct 循环，并行工具，熔断保护 |
| **三层记忆** | `src/memory/` | SQLite(L1) + Postgres(L2) + pgvector/BM25(L3) |
| **四层压缩** | `src/context/` | Budget截断 → 冗余裁剪 → 精缩 → 自适应 |
| **Skill 进化** | `src/skills/` | 双层沉淀 + 版本管理 + 成功率追踪 |

### 记忆架构

```
L1 会话态 (SQLite)       ← 当前对话，即时读写
L2 工作记忆 (Postgres)    ← 用户画像、偏好，跨会话保留
L3 长期记忆 (pgvector)    ← Dense + BM25 + RRF 混合检索
```

### 工具列表

| 工具 | 功能 |
|------|------|
| `file_read/write/edit` | 文件操作（原子写入 + 备份） |
| `calculator` | 安全计算器 |
| `web_search` | DuckDuckGo 搜索 |
| `web_fetch` | 网页抓取 |
| `memory_write` | 写入记忆 |
| `memory_recall` | 召回记忆 |
| `skill_invoke` | 加载技能 |
| `task_manager` | 任务管理 |
| `finish` | 结束信号 |

## 配置

```bash
# .env
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com

PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=clawcore
PG_USER=postgres
PG_PASSWORD=clawcore123

EMBEDDING_MODEL=BAAI/bge-m3
MAX_STEPS=10
```

## 项目结构

```
ClawCore/
├── config/settings.py          # 全局配置（6组 dataclass）
├── src/
│   ├── cli.py                  # CLI 入口
│   ├── agent_loop/             # Agent 引擎
│   │   └── react_loop.py       # ReAct 主循环
│   ├── memory/                 # 记忆系统
│   │   ├── session_memory.py   # L1 SQLite
│   │   ├── working_memory.py   # L2 Postgres
│   │   ├── long_term_memory.py # L3 pgvector+BM25
│   │   ├── memory_manager.py   # 三层统一入口
│   │   └── memory_consolidator.py # 沉淀调度
│   ├── context/                # 压缩系统
│   │   ├── context_pipeline.py # 四层编排
│   │   ├── budget_truncator.py # L1 截断
│   │   ├── redundancy_pruner.py # L2 去重
│   │   ├── structural_compressor.py # L3 摘要
│   │   └── auto_threshold.py   # L4 自适应
│   ├── skills/                 # Skill 系统
│   │   ├── skill_manager.py    # 生命周期管理
│   │   ├── skill_extractor.py  # 模式挖掘
│   │   ├── skill_validator.py  # 自动验证
│   │   └── skill_versioning.py # 版本管理
│   ├── tools/                  # 9 个工具
│   └── llm/                    # LLM 适配
└── data/                       # 运行时数据
```

## 技术栈

Python 3.10+ | DeepSeek | Postgres + pgvector | BGE-M3 | BM25 + jieba | Docker | SQLite | Typer
