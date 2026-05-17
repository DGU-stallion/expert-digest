# Zhihu (知乎) Crawler

平台标识：`zhihu`

从知乎用户主页抓取回答（answers）和文章（articles），规范化后导入 ExpertDigest 处理管线。

**重要**：Agent 通过 Playwright MCP Bridge 在已登录的 Chrome 浏览器中执行 API 请求，利用浏览器已有的登录会话完成抓取。

---

## 前置条件

1. **Chrome 浏览器已登录 [zhihu.com](https://www.zhihu.com)** — 这是唯一支持的登录方式
2. **Playwright MCP Bridge 扩展** 已安装在 Chrome 中
3. Claude Code Agent 可正常调用 `browser_navigate`、`browser_evaluate` 等 Playwright 工具

---

## 架构

```
Agent（Claude Code + MCP Playwright）
 │
 ├── 1. browser_navigate → zhihu.com（确认已登录）
 ├── 2. browser_evaluate → 在浏览器中执行 fetch()
 │      使用 credentials: 'include' 自动携带登录态
 ├── 3. 保存原始 JSON 到临时文件
 │
 └── 4. expert-digest ingest-agent-data zhihu \
          --user-token <token> \
          --profile-file <file> \
          --answers-file <file> \
          --articles-file <file>
          │
          ├── 调用 node src/agent-crawl.js 规范化数据
          └── 自动执行 import-crawler 导入 SQLite
```

整个流程在 Agent（Claude Code）中一步完成。Node.js 模块只负责数据处理（规范化、文件 I/O），不负责数据获取。

---

## 两种运行模式

### 1. Agent 驱动（推荐）

Agent 通过 Playwright MCP 工具在用户的 Chrome 中执行 API 请求：

```bash
# Agent 执行以下步骤（用户在 CLAUDE.md 中已定义）：
# 1. browser_navigate → https://www.zhihu.com
# 2. browser_evaluate → 获取 profile
# 3. browser_evaluate → 获取 answers
# 4. browser_evaluate → 获取 articles
# 5. 保存临时文件
# 6. expert-digest ingest-agent-data zhihu ...
```

具体步骤见 `.claude/CLAUDE.md` 中的 `## 爬虫工作流` 章节。

### 2. 直接使用 Node.js 处理已获取的数据

如果 Agent 已通过其他方式获取了原始 JSON 文件，可手动执行：

```bash
cd crawlers/zhihu
node src/agent-crawl.js \
  --user-token huang-wei-yan-30 \
  --output-dir ../../data/crawlers/zhihu \
  --profile-file /tmp/profile.json \
  --answers-file /tmp/answers.json \
  --articles-file /tmp/articles.json
```

---

## 输出目录结构

```
data/crawlers/zhihu/<user-token>/
├── raw/
│   ├── profile.json          # 用户资料（原始）
│   ├── answers.jsonl         # 回答（原始）
│   └── articles.jsonl        # 文章（原始）
├── index/
│   └── content_index.jsonl   # 规范化后的索引（含 platform 字段）
└── logs/
    └── crawl.log             # 处理日志
```

`content_index.jsonl` 遵循 `crawlers/shared/schema.md` 规范，每条记录包含 `platform: "zhihu"` 字段供加载器分发。

---

## 测试

```bash
cd crawlers/zhihu
npm test
```
