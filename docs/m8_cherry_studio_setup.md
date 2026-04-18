# M8 Cherry Studio MCP 接入说明

更新时间：2026-04-18

本文用于把 `ExpertDigest` 的 MCP baseline 接入 Cherry Studio。

## 1. 前置条件

1. 已拉取仓库到本地：`D:\Project\Expert_Digest`
2. Python 3.11+
3. 已创建并可用 `.venv`
4. 本地已有可查询数据库（例如：`data/processed/zhihu_huang.sqlite3`）
5. 若使用 hybrid/LLM 主题总结，建议在 Cherry Studio Provider 中配置 Google API（`gemini-2.5-flash`）

## 2. 安装 MCP 依赖

在仓库根目录执行：

```powershell
cd D:\Project\Expert_Digest
.\.venv\Scripts\python.exe -m pip install -e ".[mcp]"
```

## 3. 本地启动 MCP Server

推荐用 `stdio`（Cherry Studio 最常见）：

```powershell
cd D:\Project\Expert_Digest
.\.venv\Scripts\expert-digest.exe run-mcp-server --db data/processed/zhihu_huang.sqlite3 --transport stdio
```

如果要验证 `sse` 参数链路：

```powershell
.\.venv\Scripts\expert-digest.exe run-mcp-server --db data/processed/zhihu_huang.sqlite3 --transport sse
```

## 4. 在 Cherry Studio 中配置 MCP

在 Cherry Studio 的 MCP 配置里新增一个本地服务（示意）：

- 名称：`expert-digest`
- 启动命令：`D:\Project\Expert_Digest\.venv\Scripts\expert-digest.exe`
- 启动参数：`run-mcp-server --db data/processed/zhihu_huang.sqlite3 --transport stdio`
- 工作目录：`D:\Project\Expert_Digest`

如果 Cherry Studio 支持 JSON 配置，核心就是同一条命令与参数。

LLM 相关命令（例如 `generate-handbook --synthesis-mode hybrid` / `cluster-topics --label-mode llm`）默认读取：

`C:\Users\<你自己的用户名>\.cc-switch\cc-switch.db`

并优先尝试 Google/Gemini provider。

## 5. 工具列表（M8 baseline）

已暴露工具：

1. `ask_author`
2. `search_posts`
3. `recommend_readings`
4. `list_topics`
5. `generate_handbook`
6. `generate_skill`

## 6. 冒烟测试建议

在 Cherry Studio 内依次调用：

1. `ask_author(question="长期主义在项目管理里的核心是什么？")`
2. `search_posts(query="反馈密度", top_k=3)`
3. `list_topics(num_topics=3, top_docs=2)`
4. `generate_handbook(output_path="data/outputs/cherry_handbook.md")`
5. `generate_skill(output_path="data/outputs/cherry_skill.md")`

预期：

1. 问答返回 `answer/evidence/recommended_original`
2. 搜索返回 `hits`
3. 主题返回 `topics`
4. 手册文件生成到指定路径
5. Skill 文件生成到指定路径

## 7. 常见问题

### Q1: `Failed to start MCP server: MCP dependency is missing`

执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[mcp]"
```

### Q2: 工具可见但调用失败（找不到数据库）

确认 Cherry Studio 的工作目录是仓库根目录，或把 `--db` 改成绝对路径。

### Q3: 数据库有文档但检索为空

先执行：

```powershell
.\.venv\Scripts\expert-digest.exe rebuild-chunks --db data/processed/zhihu_huang.sqlite3
.\.venv\Scripts\expert-digest.exe rebuild-embeddings --db data/processed/zhihu_huang.sqlite3
```

### Q4: hybrid 模式没有走 Gemini

1. 确认 provider DB 中有 `Google/Gemini` 可用配置（含 API Key）。  
2. 检查命令是否使用了正确配置库路径：`--llm-config-db <path-to-db>`（兼容旧参数 `--ccswitch-db`）。  
3. 用 `generate-handbook --format json` 查看 `llm_provider/llm_model/llm_base_url` 字段确认运行时实际选中的 provider。

## 8. 验收清单

1. MCP 服务可启动
2. Cherry Studio 能看到 6 个工具
3. `ask_author` 与 `search_posts` 有返回结果
4. `generate_handbook` / `generate_skill` 能落地文件
