# ExpertDigest Wiki Foundation Real Data Validation

Date: 2026-04-21

## 验证目标

使用真实黄彦臻 Zhihu 数据库验证 Wiki Foundation MVP 是否能在 824 篇文章上完成端到端链路：

```text
SQLite source documents -> hierarchical evidence -> Markdown wiki vault -> search -> quality report
```

## 执行命令

```powershell
.\.venv\Scripts\expert-digest.exe build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild
.\.venv\Scripts\expert-digest.exe build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang --expert-id huang --expert-name "黄彦臻" --purpose "沉淀黄彦臻公开文章中的投资分析框架。"
.\.venv\Scripts\expert-digest.exe search-wiki "泡泡玛特 核心能力" --wiki-root data/wiki/huang
.\.venv\Scripts\expert-digest.exe eval-wiki --wiki-root data/wiki/huang --expected-source-count 824
```

## 结果指标

- Source documents: 824
- Parent sections: 830
- Child chunks: 4744
- Evidence spans: 8055
- Source pages: 824
- Topic pages: 2866
- Concept pages: 5640
- Total evaluated wiki pages: 9330
- Markdown files in vault: 9334
- Vault Markdown size: 13,815,057 bytes
- `index.md` size: 749,251 bytes
- `log.md` size: 134,844 bytes

Quality report:

```json
{
  "page_count": 9330,
  "source_page_count": 824,
  "pages_with_sources": 9330,
  "pages_missing_sources": [],
  "traceability_ratio": 1.0,
  "coverage_ratio": 1.0
}
```

## 抽检结论

### 通过项

- 全量 824 篇真实文章可以完成 evidence 构建和 wiki 写入。
- 每篇 source 都生成了 `sources/<source_id>.md`。
- 所有被评估页面都有 source refs，traceability 达到 1.0。
- `search-wiki "泡泡玛特 核心能力"` 能返回相关 source/concept/topic 页面。
- Source 页面基本可读，包含摘要、核心判断和 evidence span ids。

### 主要问题

1. Concept/topic 过碎

当前 deterministic analyzer 直接从标题和正文 token 中抽取候选概念，真实数据下生成了 5640 个 concept 和 2866 个 topic。抽检发现大量页面是句子片段、标题片段或低信息词，例如：

- `按他们`
- `日午间`
- `全市场逾`
- `只个股涨停`
- `泡泡玛特登顶美国`
- `农耕文明的天性就是惧怕风险所延伸出基于朴素唯物主义`

2. Index 页面不可维护

`index.md` 采用线性追加方式，真实数据下达到 749 KB。它证明链路可用，但已经不适合作为人工导航入口。

3. Concept/topic 页面合并策略不足

同名 slug 页面会被后续写入覆盖，页面不能稳定累积多个来源的判断。当前 source refs 覆盖率为 1.0，但概念页还不是可靠的跨文章知识汇总。

4. 检索性能偏慢

`search-wiki` 和 `eval-wiki` 都需要扫描并解析数千个 Markdown 文件。真实 vault 下命令可用，但一次查询约为分钟级，后续需要缓存、索引或轻量 SQLite metadata。

5. Source 摘要仍是 baseline

Source 摘要取自第一条核心 evidence span，忠实但不一定是文章真正摘要。它适合作为 MVP 证明，不适合作为最终 handbook/skill 输入质量。

## 质量判断

Wiki Foundation MVP 已通过工程链路验收：

- 数据完整性：通过
- 来源回溯：通过
- CLI 可用性：通过
- Markdown vault 生成：通过

但尚未通过知识质量验收：

- Concept/topic 命名质量：需要改进
- Wiki 页面聚合能力：需要改进
- 人工导航可用性：需要改进
- 检索性能：需要改进

## 推荐下一阶段任务

1. 改进 deterministic analyzer
   - 增加中文停用词和问题模板过滤。
   - 过滤过短、过长、含日期/数字噪声的概念。
   - 对标题中的问题句做主题提纯。
   - 区分 entity、concept、topic。

2. 改进 wiki writer 聚合策略
   - 写 concept/topic 时读取已有页面并追加 source refs。
   - 同一 concept/topic 聚合多个 source 的 claims。
   - 避免后续 source 覆盖已有概念页。

3. 改进 index 与导航
   - 将 `index.md` 改为摘要入口。
   - 生成独立 `sources/index.md`、`topics/index.md`、`concepts/index.md`。
   - 增加高频 concept/topic 排行，而不是线性列出每篇文章。

4. 增加 wiki lint
   - 标记过短/过长标题。
   - 标记低信息 concept/topic。
   - 标记孤立页、重复页、被覆盖风险。
   - 输出可行动的修复建议。

5. 再考虑 LLM refinement
   - LLM 可用于概念合并、标题改写、thesis 候选生成。
   - LLM 不应接管 source refs 和 evidence traceability。

## 当前建议

先不要进入 handbook/skill 重写。下一步应先做 Wiki Quality Pass 1：

```text
analyzer 过滤与归一化 -> writer 聚合 -> index 分层 -> wiki lint
```

完成后再重新跑 824 篇数据，并比较 concept/topic 数量、搜索耗时和抽检质量。

## 2026-04-22 Wiki Quality Pass 1 复跑（对比基线）

### 复跑说明

- 为避免历史目录残留影响统计，本次使用全新目录：`data/wiki/huang_pass1`。
- 数据源保持不变：`data/processed/zhihu_huang.sqlite3`（824 篇）。

### 执行命令

```powershell
.\.venv\Scripts\expert-digest.exe build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild
.\.venv\Scripts\expert-digest.exe build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang_pass1 --expert-id huang --expert-name "黄彦臻" --purpose "沉淀黄彦臻公开文章中的投资分析框架。"
.\.venv\Scripts\expert-digest.exe search-wiki "泡泡玛特 核心能力" --wiki-root data/wiki/huang_pass1
.\.venv\Scripts\expert-digest.exe eval-wiki --wiki-root data/wiki/huang_pass1 --expected-source-count 824
```

### 结果指标（Pass 1）

- Source documents: 824
- Parent sections: 830
- Child chunks: 4744
- Evidence spans: 8055
- Source pages: 824
- Topic pages: 2961
- Concept pages: 5757
- Total evaluated wiki pages: 9542
- Markdown files in vault: 9549
- Vault Markdown size: 15,199,672 bytes
- `index.md` size: 192 bytes
- `sources/index.md` size: 124,152 bytes
- `topics/index.md` size: 211,195 bytes
- `concepts/index.md` size: 412,227 bytes
- `log.md` size: 134,844 bytes

Quality report:

```json
{
  "page_count": 9542,
  "source_page_count": 824,
  "pages_with_sources": 9542,
  "pages_missing_sources": [],
  "traceability_ratio": 1.0,
  "coverage_ratio": 1.0
}
```

### 与 2026-04-21 基线对比

| 指标 | 2026-04-21 基线 | 2026-04-22 Pass 1 | 变化 |
| --- | ---:| ---:| ---:|
| Source pages | 824 | 824 | 0 |
| Topic pages | 2866 | 2961 | +95 |
| Concept pages | 5640 | 5757 | +117 |
| Evaluated wiki pages | 9330 | 9542 | +212 |
| `traceability_ratio` | 1.0 | 1.0 | 0 |
| `coverage_ratio` | 1.0 | 1.0 | 0 |
| 根 `index.md` 大小 | 749,251 B | 192 B | -749,059 B |

### 本轮结论

1. 导航结构目标达成：根 `index.md` 从日志式大文件收敛为总览入口，分层索引生效。
2. 可追溯性目标保持：`traceability_ratio=1.0`、`coverage_ratio=1.0`。
3. 数量收敛目标未达成：concept/topic 数量较基线仍有小幅上升。

### 后续建议

下一轮应继续收紧 analyzer 规则（尤其是标题归一化与低信息短语过滤），并增加针对真实噪声样本的回归测试，直到 concept/topic 总量相对基线出现稳定下降。
