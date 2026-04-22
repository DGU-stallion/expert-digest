# ExpertDigest Wiki Foundation Usage

## 目标

Wiki Foundation 把专家原文编译成 Markdown vault。它是 handbook、skill 和问答升级的中间知识层。

## 推荐流程

```powershell
expert-digest import-zhihu "D:\Project\Zhihu_Crawler\data\zhihu\huang-wei-yan-30" --db data/processed/zhihu_huang.sqlite3
expert-digest build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild
expert-digest build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang --expert-id huang --expert-name "黄彦臻" --purpose "沉淀黄彦臻公开文章中的投资分析框架。"
expert-digest search-wiki "泡泡玛特 核心能力" --wiki-root data/wiki/huang
expert-digest eval-wiki --wiki-root data/wiki/huang --expected-source-count 824
```

## Vault 结构

```text
data/wiki/<expert_id>/
  purpose.md
  schema.md
  index.md
  log.md
  sources/
  concepts/
  topics/
  theses/
  reviews/
```

## 质量标准

- 每个 source 都应有 `sources/<source_id>.md`。
- 每个生成页面应有 frontmatter。
- 核心页面应带 source refs。
- `eval-wiki` 应输出 traceability 和 coverage。
