# ExpertDigest — 专家内容知识蒸馏工具

将领域专家的公开文章转化为结构化学习手册（Handbook）和技能描述文件（SKILL），实现个人知识的系统化沉淀与复用。

---

## 声明：爬虫模块

**本仓库不包含爬虫模块。** 所有数据均来自预先爬取好的结构化文件，支持以下三种导入格式：

| 格式 | 说明 |
|---|---|
| JSONL | 每行一条 JSON 对象，含 `author`、`title`、`content`、`source` 等字段 |
| Markdown | 带 YAML 头部的 Markdown 文件 |
| 知乎导出 | 知乎爬虫工具导出的 `index/content_index.jsonl` 格式 |

如需爬取数据，请使用独立的爬虫项目（如知乎爬虫、公众号采集等），将结果导出为上述格式后再导入 ExpertDigest。

---

## 设计架构

```
                         ┌──────────────────────┐
                         │    JSONL / Markdown   │
                         │    / Zhihu 导出数据    │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │     SQLite 存储       │
                         │  (documents / chunks  │
                         │   embeddings / spans) │
                         └──────────┬───────────┘
                                    ▼
              ┌─────────────────────┴─────────────────────┐
              │                                           │
              ▼                                           ▼
   ┌─────────────────────┐                   ┌─────────────────────┐
   │  主题聚类 + 向量检索  │                   │   Wiki 知识库        │
   │  (community detect)  │                   │   (Markdown Vault)   │
   └──────────┬──────────┘                   └──────────┬──────────┘
              │                                         │
              ▼                                         ▼
   ┌─────────────────────────────────────────────────────────┐
   │              LangGraph 蒸馏管线 (Pipeline)                │
   │                                                         │
   │   入口 → 主题聚类 → LLM 分析 → 质量检查                    │
   │               ↓                                         │
   │     ┌─────────────────┐   ┌─────────────────┐           │
   │     │  Handbook 子图    │   │  SKILL 子图      │           │
   │     │  (规划→撰写→评审  │   │  (心智模型→编码   │           │
   │     │   →编辑→追踪)     │   │   →协议→组装)    │           │
   │     └────────┬────────┘   └────────┬────────┘           │
   │              │                     │                     │
   └──────────────┼─────────────────────┼─────────────────────┘
                  ▼                     ▼
         ┌──────────────┐     ┌──────────────┐
         │ handbook.md  │     │   skill.md   │
         │ (学习手册)     │     │  (技能描述)   │
         └──────────────┘     └──────────────┘
```

### 核心组件

| 模块 | 路径 | 职责 |
|---|---|---|
| **数据导入** | `src/expert_digest/ingest/` | 加载 JSONL / Markdown / 知乎导出文件到 SQLite |
| **数据加工** | `src/expert_digest/processing/` | 文档清洗、分块、哈希词袋向量化、证据层级构建 |
| **向量存储** | `src/expert_digest/storage/` | SQLite 持久化（5 张表），确定性 ID（SHA-256） |
| **语义检索** | `src/expert_digest/retrieval/` | 余弦相似度排序，Top-K 检索 |
| **RAG 问答** | `src/expert_digest/rag/` | 结构化问答输出，含置信度评分与拒绝策略 |
| **知识分析** | `src/expert_digest/knowledge/` | 主题聚类（社区检测 + 质心聚类）、作者画像 |
| **Wiki 知识库** | `src/expert_digest/wiki/` | Markdown Vault 抽象层，页面生成/评估/检索 |
| **蒸馏管线** | `src/expert_digest/pipeline/` | LangGraph StateGraph 编排 Handbook 和 SKILL 生成 |
| **MCP 服务** | `src/expert_digest/mcp/` | Model Context Protocol 服务，用于 Cherry Studio 集成 |
| **Streamlit 应用** | `src/expert_digest/app/` | 本地演示 UI（非主要产品路径） |

### 技术栈

- **运行环境**：Python 3.11+
- **编排引擎**：LangGraph（管线状态机）
- **LLM API**：DeepSeek（通过 Anthropic 兼容接口），分 fast / reasoning 两个档位
- **向量化**：确定性哈希词袋（无需外部嵌入模型）
- **外部依赖**：零核心依赖，可选安装 `[pipeline]`、`[mcp]`、`[app]` 分组

---

## 快速开始

### 1. 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

安装完整功能（推荐）：

```powershell
python -m pip install -e ".[dev,pipeline,mcp]"
```

### 2. 配置 API Key

```powershell
# 复制模板，填入你的 DeepSeek API Key
cp .env.example .env
# 编辑 .env，填入真实密钥
```

### 3. 验证安装

```powershell
python -m pytest
python -c "import expert_digest; print(expert_digest.__version__)"
```

---

## 数据准备

ExpertDigest 使用预先爬取好的结构化数据，不支持直接爬取。

### 导入 JSONL

```powershell
expert-digest import-jsonl data/sample/articles.jsonl --db data/processed/expert_digest.sqlite3
```

JSONL 每行格式：

```json
{"author":"黄彦臻","title":"标题","content":"正文...","source":"zhihu:answer:123","url":"https://...","created_at":"2026-04-03T07:55:11.000Z"}
```

必填字段：`author`、`title`、`content`、`source`
可选字段：`url`、`created_at`

### 导入 Markdown

```powershell
expert-digest import-markdown path/to/markdown-folder --db data/processed/expert_digest.sqlite3
```

支持 YAML 头部：

```markdown
---
author: 黄彦臻
title: 标题
url: https://...
created_at: 2026-04-03T07:55:11.000Z
---

正文内容。
```

### 导入知乎导出

```powershell
expert-digest import-zhihu "D:\data\zhihu\export" --db data/processed/zhihu.sqlite3
```

---

## 处理管线

导入数据后，执行以下步骤构建知识索引：

```powershell
# 1. 文档分块
expert-digest build-chunks --db data/processed/zhihu.sqlite3

# 2. 生成向量（确定性哈希词袋，无需外部模型）
expert-digest build-embeddings --db data/processed/zhihu.sqlite3

# 3. 构建证据层级（多级上下文结构）
expert-digest build-evidence --db data/processed/zhihu.sqlite3 --rebuild

# 4. 构建 Wiki 知识库（可选）
expert-digest build-wiki --db data/processed/zhihu.sqlite3 --wiki-root data/wiki/default --expert-id huang --expert-name "黄彦臻"
```

---

## 知识蒸馏

这是 ExpertDigest 的核心功能：通过 LLM 驱动的 LangGraph 管线，将作者的知识体系蒸馏为结构化文档。

### 生成学习手册（Handbook）

```powershell
expert-digest generate-handbook-pipeline --db data/processed/zhihu_huang.sqlite3 --author 黄彦臻
```

输出：`data/outputs/handbook.md`

管线自动执行以下步骤：
1. **主题聚类** — 基于向量相似性发现核心主题
2. **LLM 分析** — 提取核心主题、关键概念、思维模式、表达风格
3. **章节规划** — 设计学习路径，规划 5-10 个章节
4. **章节撰写** — 基于原文逐章生成，含引用标注
5. **质量评审** — 事实依据、结构完整性检查，支持重写循环
6. **连贯性编辑** — 去重、术语统一、逻辑流优化、最终润色

### 生成技能描述（SKILL）

```powershell
expert-digest generate-skill-pipeline --db data/processed/zhihu_huang.sqlite3 --author 黄彦臻
```

输出：`data/outputs/skill.md`

SKILL 文件包含：
- **角色扮演规则** — AI 模仿作者语气和思维方式的规则集
- **身份卡 / 表达 DNA** — 作者背景和风格描述
- **回答工作流（Agentic Protocol）** — 问题分类、研究维度、回答框架
- **核心心智模型** — 作者反复使用的分析框架（如波浪周期、人口-资产映射、高股息价值金字塔等）
- **决策启发式** — 经验法则
- **价值观与反模式** — 核心理念 vs 明确反对的做法
- **诚实边界** — 作者自知的知识/能力边界

---

## 问答 & RAG

基于向量检索的结构化问答：

```powershell
expert-digest ask "泡泡玛特的核心能力是什么？" --db data/processed/zhihu_huang.sqlite3 --top-k 3
```

支持 JSON 格式输出：

```powershell
expert-digest ask "泡泡玛特的核心能力是什么？" --db data/processed/zhihu_huang.sqlite3 --format json
```

输出包含：回答、依据（含来源和评分）、推荐原文、不确定性说明。

---

## MCP 服务

启动 Model Context Protocol 服务，用于 Cherry Studio 等 MCP 客户端集成：

```powershell
python -m pip install -e ".[mcp]"
expert-digest run-mcp-server --db data/processed/zhihu_huang.sqlite3 --transport stdio
```

详见：`docs/m8_cherry_studio_setup.md`

---

## 完整工作流示例

从零开始到生成 Handbook + SKILL 的一站式流程：

```powershell
# 1. 导入数据
expert-digest import-jsonl data/sample/articles.jsonl --db data/processed/expert_digest.sqlite3

# 2. 构建分块和向量
expert-digest build-chunks --db data/processed/expert_digest.sqlite3
expert-digest build-embeddings --db data/processed/expert_digest.sqlite3

# 3. 蒸馏知识
expert-digest generate-handbook-pipeline --db data/processed/expert_digest.sqlite3 --author 黄彦臻
expert-digest generate-skill-pipeline --db data/processed/expert_digest.sqlite3 --author 黄彦臻

# 查看结果
cat data/outputs/handbook.md
cat data/outputs/skill.md
```

项目提供了自动化验证脚本：

```powershell
.\scripts\quickstart_selfcheck.ps1
```

---

## 项目结构

```
├── src/expert_digest/
│   ├── cli.py                    # 命令行入口（18+ 个子命令）
│   ├── config.py                 # 路径配置
│   │
│   ├── domain/models.py          # 核心数据模型
│   ├── ingest/                   # 数据导入（JSONL / Markdown / 知乎）
│   ├── processing/               # 清洗、分块、向量化、证据构建
│   ├── storage/sqlite_store.py   # SQLite 持久化层
│   ├── retrieval/                # 余弦相似度检索
│   ├── rag/                      # 结构化 RAG 问答
│   ├── knowledge/                # 主题聚类、作者画像
│   ├── wiki/                     # Markdown Wiki 知识库
│   │
│   ├── pipeline/                 # LangGraph 蒸馏管线
│   │   ├── graph.py              # 主图编排
│   │   ├── state.py              # 管线状态定义
│   │   ├── llm.py                # LLM 客户端工厂
│   │   ├── nodes/                # 入口、聚类、分析、表达、质量节点
│   │   ├── handbook/             # Handbook 子图（规划、撰写、评审、编辑、追踪）
│   │   └── skill/                # SKILL 子图（心智模型、编码、协议、组装、验证）
│   │
│   ├── generation/llm_client.py  # Anthropic 兼容 HTTP 客户端
│   ├── mcp/                      # Model Context Protocol 服务
│   └── app/                      # Streamlit 演示 UI
│
├── tests/                        # pytest 测试套件
├── configs/                      # YAML/JSON 配置文件
├── data/                         # 运行时数据
│   ├── sample/                   # 示例数据（30 条测试文章）
│   └── processed/                # 处理后数据库（gitignored）
│
├── docs/                         # 设计文档和里程碑记录
├── scripts/                      # 辅助脚本
└── .env.example                  # 环境变量模板
```

---

## 开发

```powershell
# 运行测试
python -m pytest

# 代码检查
python -m ruff check .
```

---

## 许可

本项目采用 MIT 许可证。
