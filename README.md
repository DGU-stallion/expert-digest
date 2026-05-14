# ExpertDigest — 专家内容知识蒸馏引擎

将领域专家的公开文章转化为结构化学习手册（Handbook）和技能描述文件（SKILL），供 Claude Code / Cursor / MCP 等 AI Agent 直接使用。

---

## 项目用途

ExpertDigest 解决了一个核心问题：**如何让 AI 真正“学会”一个专家的知识体系和思考方式？**

- **Handbook** — 完整学习手册，让 AI 快速理解专家的知识框架
- **SKILL.md** — Agent 专用技能文件，让 AI 模仿专家的语气、逻辑和决策模式
- **Wiki Vault** — 可检索的知识图谱，概念+来源双向追溯

输出的 `handbook.md` 和 `skill.md` 可直接用于 Claude Code 的 `@memory` 或 MCP 服务。

---

## 快速开始

### 1. 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,pipeline]"
```

### 2. 准备数据

三种导入格式（任选其一）：

```powershell
# JSONL 导入（推荐）
expert-digest import-jsonl data/sample/articles.jsonl --db data/processed/db.sqlite3

# Markdown 导入
expert-digest import-markdown path/to/markdown/ --db data/processed/db.sqlite3

# 知乎导出导入
expert-digest import-zhihu path/to/zhihu-export --db data/processed/db.sqlite3
```

### 3. 构建索引

```powershell
# 分块 + 向量化
expert-digest build-chunks --db data/processed/db.sqlite3
expert-digest build-embeddings --db data/processed/db.sqlite3

# （可选）构建 Wiki 知识库
expert-digest build-wiki --db data/processed/db.sqlite3 --wiki-root data/wiki/default
```

### 4. 知识蒸馏

```powershell
# 生成学习手册（~5-10分钟，LLM 驱动）
expert-digest generate-handbook --db data/processed/db.sqlite3 --author 作者名

# 生成技能描述（SKILL.md）
expert-digest generate-skill --db data/processed/db.sqlite3 --author 作者名
```

### 5. 在 Claude Code 中使用

```powershell
# 将输出文件加入 Claude 的 memory
claude memory add data/outputs/handbook.md --role handbook
claude memory add data/outputs/skill.md --role skill
```

或者在 `.claude/memory.json` 中配置自动加载。

---

## 项目架构

```
                      ┌────────────────────┐
                      │   输入数据层        │
                      │  JSONL / Markdown  │
                      │    / 知乎导出       │
                      └────────┬───────────┘
                               ▼
                      ┌────────────────────┐
                      │   存储层: SQLite    │
                      │  documents/chunks   │
                      │    embeddings       │
                      └────────┬───────────┘
                               ▼
┌───────────────────────┐     ┴     ┌───────────────────────┐
│   分析阶段（共享）    │──────────▶│     Wiki 知识库        │
│                       │           │  概念/主题/来源页      │
│  聚类 → 分析 → 评估  │           │   双向追溯索引         │
└──────────────┬────────┘           └───────────────────────┘
               │
               ▼
    ┌──────────┴──────────┐
    │  管线分流 (LangGraph)│
    └──────────┬──────────┘
               │
     ┌─────────┴──────────┐
     ▼                    ▼
┌──────────┐       ┌──────────┐
│ Handbook │       │  SKILL   │
│  子图    │       │   子图   │
└──┬───┬──┘       └──┬───┬──┘
   │   │             │   │
   │   └─────────────┘   └────────────────┐
   │                                       │
   ▼                                       ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│ 章节规划 → 撰写 → 评审 → 编辑│  │ 心智模型 → 表达编码 → 协议  │
│  + 增量磁盘缓存 + 选择性重写│  │  + 质量验证                  │
└──────────────┬──────────────┘   └──────────────┬──────────────┘
               │                                   │
               ▼                                   ▼
     ┌───────────────────┐              ┌───────────────────┐
     │  handbook.md      │              │    skill.md       │
     │  (学习手册)       │              │   (技能描述)      │
     └───────────────────┘              └───────────────────┘
```

---

## 核心流程

### 分析阶段

```
文档加载 → 主题聚类 → LLM内容分析 → LLM表达分析 → 质量评估
                                                           │
                                                      失败 ◄┘
                                                           │
                                                      通过 → 分叉
```

### Handbook 子图

```
章节规划 → [撰写 → 评审]ⁿ → 连贯性编辑 → 引用追踪 → handbook.md
           ↑_____________↓
        失败则重写（仅重写未通过章节）
```

**特性：**
- 增量磁盘缓存：每章写入 `.handbook_cache/`，支持断点续跑
- 选择性重写：仅重写评审失败的章节
- 引用追踪：所有观点关联到原文来源

### SKILL 子图

```
心智模型提取 → 表达编码 → 协议设计 → SKILL组装 → 验证 → skill.md
```

**SKILL 包含：**
- 角色扮演规则（语气、风格、思维模式）
- 表达 DNA（常用句式、确信光谱、引用习惯）
- 智能体协议（问题分类 → 研究维度 → 回答框架）
- 核心心智模型
- 决策启发式
- 价值观与反模式

---

## 项目结构

```
expert-digest/
├── src/expert_digest/
│   ├── cli.py                          # 命令行入口
│   ├── domain/models.py                # 核心数据模型
│   │
│   ├── ingest/                         # 数据导入模块
│   │   ├── jsonl_loader.py
│   │   ├── markdown_loader.py
│   │   └── zhihu_loader.py
│   │
│   ├── processing/                     # 数据处理
│   │   ├── cleaner.py                  # 文本清洗
│   │   ├── splitter.py                 # 智能分块
│   │   └── embedder.py                 # 哈希词袋向量化
│   │
│   ├── storage/sqlite_store.py         # SQLite 持久化
│   ├── retrieval/retriever.py          # 余弦相似度检索
│   │
│   ├── knowledge/                      # 知识分析
│   │   ├── community_detection.py      # 社区检测算法
│   │   ├── topic_clusterer.py          # 主题聚类核心
│   │   └── topic_report.py             # 聚类质量报告
│   │
│   ├── generation/llm_client.py        # LLM 客户端（含重试逻辑）
│   │
│   ├── wiki/                           # Wiki Vault 知识库
│   │   ├── vault.py                    # 文件系统抽象
│   │   ├── writer.py                   # 分析结果写入
│   │   ├── analyzer.py                 # 文档级分析
│   │   ├── evaluator.py                # 覆盖率评估
│   │   └── linter.py                   # 孤立页面检测
│   │
│   └── pipeline/                       # LangGraph 蒸馏管线
│       ├── graph.py                    # 主编排
│       ├── state.py                    # 状态定义
│       ├── llm.py                      # LLM 工厂
│       ├── nodes/                      # 分析节点
│       ├── handbook/                   # Handbook 子图
│       └── skill/                      # SKILL 子图
│
├── tests/                              # pytest 测试套件
├── configs/                            # 配置文件
├── data/
│   ├── sample/                         # 示例数据
│   ├── processed/                      # SQLite 数据库
│   ├── wiki/                           # Wiki Vault
│   └── outputs/                        # 输出产物 (handbook.md, skill.md)
│
├── .env.example                        # 环境变量模板
├── pyproject.toml
└── README.md
```

---

## 技术栈

- **编排引擎**：LangGraph (StateGraph)
- **LLM**：DeepSeek / OpenAI 兼容接口（支持 fast + reasoning 双档位）
- **向量化**：确定性哈希词袋 Bag-of-Words（零外部依赖）
- **存储**：SQLite
- **测试**：pytest

---

## 示例输出

- `data/outputs/handbook.md` — 黄彦臻投资学习手册（56KB，7章）
- `data/outputs/skill.md` — 黄彦臻投资技能（18KB，Agent 专用）

---

## License

MIT
