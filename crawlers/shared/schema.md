# Crawler Output Schema

所有平台爬虫输出的 `content_index.jsonl` 需遵循以下规范，才能被 Expert_Digest 的 `import-crawler` 命令正确导入。

## 核心字段

```jsonc
{
  // ---- 必填字段 ----
  "platform": "zhihu",               // 平台标识，loader 分发 key
  "source_type": "answer",           // 内容类型（平台特有）
  "source_id": "12345",              // 内容唯一 ID
  "author_name": "黄彦臻",           // 作者名
  "title": "文章标题",               // 内容标题
  "url": "https://...",              // 原文链接

  // ---- 内容（三者至少提供一个） ----
  "content_text": "纯文本正文",       // 纯文本（推荐）
  "content_markdown": "# Markdown",  // Markdown 格式
  "content_html": "<p>HTML</p>",     // HTML 格式

  // ---- 时间戳 ----
  "created_at": "2026-04-09T07:30:19.000Z",  // ISO 8601
  "updated_at": "2026-04-09T07:30:19.000Z",

  // ---- 平台特有扩展字段 ----
  "platform_specific": {
    "question_id": "...",
    "voteup_count": 100,
    // 每个平台可自由扩展
  }
}
```

## 文件布局

```
data/crawlers/<platform>/
  <user_token>/
    index/
      content_index.jsonl    # ← 消费就绪的输出
    raw/                     # 原始 API 响应（爬虫内部使用）
    logs/                    # 爬取日志
```

## 加载流程

```
爬虫输出 content_index.jsonl
        │
        ▼
loader.py 分发（按 platform 字段匹配）
        │
        ▼
平台 Loader（如 ZhihuLoader.load()）
        │
        ▼
list[Document] → SQLite → 后续 pipeline
```

## 当前支持的平台

| 平台 | platform 值 | Loader | 状态 |
|------|-------------|--------|------|
| 知乎 | `zhihu` | `ZhihuLoader` | 可用 |
| 雪球 | `xueqiu` | 待实现 | 占位 |
