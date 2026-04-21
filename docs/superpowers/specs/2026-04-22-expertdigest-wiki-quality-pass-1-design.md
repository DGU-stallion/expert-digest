# ExpertDigest Wiki Quality Pass 1 设计

Date: 2026-04-22

## 1. 背景与目标

Wiki Foundation MVP 已在真实 824 篇数据上打通链路，但质量验证报告显示以下关键问题：

1. `concepts/` 与 `topics/` 页面过碎，存在大量低信息标题。
2. concept/topic 页面写入是覆盖式，不能聚合多来源判断。
3. 根 `index.md` 线性追加后体积过大，不适合作为导航入口。

本次 Wiki Quality Pass 1 的目标是：在不引入 LLM、不改变核心架构的前提下，提升 Wiki 产物可读性、聚合能力和导航可维护性。

## 2. 范围与非目标

### 2.1 范围

- analyzer 过滤与归一化（deterministic）。
- writer 聚合写入策略（concept/topic 从覆盖改为累积）。
- index 分层导航（根 index 总览 + 分类型 index）。

### 2.2 非目标

- 不引入 LLM refinement。
- 不引入向量数据库或 LangGraph。
- 不重做 Streamlit/UI。
- 不实现完整 `lint-wiki` CLI（放到后续迭代）。
- 不重写 handbook/skill 生成流程。

## 3. 设计决策

### 3.1 保持 deterministic

所有规则在本地可复现：同样输入应稳定生成同样输出。避免在质量基线不稳定前引入模型不确定性。

### 3.2 先做质量收敛，再做能力扩展

本轮优先降低噪声和提高页面累积能力，而不是新增页面类型或复杂检索路径。

### 3.3 保持 CLI 入口稳定

继续使用：

- `build-evidence`
- `build-wiki`
- `search-wiki`
- `eval-wiki`

仅改变生成质量与写入行为，不更改命令语义。

## 4. 方案设计

### 4.1 Analyzer 过滤与归一化

目标：减少低信息 concept/topic。

策略：

1. 扩展停用词表，覆盖常见无主题价值词（如问句填充词、代词、泛化短语）。
2. 增加候选词过滤规则：
   - 过滤长度过短（例如 1-2 字且非稳定缩写）或过长的短语。
   - 过滤数字/日期主导短语（如“10 月 9 日”“1.56%”）作为主题词。
   - 过滤问句模板残片（如“是什么意思”“发生了什么”）。
3. 标题提纯：
   - 对问句标题做标点切分与归一化，优先保留有语义核心的词组。
4. 输出限制：
   - 保持 `concepts` 与 `topics` 数量上限，但优先级更偏向高信息词。

预期结果：concept/topic 总量下降，极短噪声页显著减少。

### 4.2 Writer 聚合写入策略

目标：让 concept/topic 页面能累积多来源证据。

策略：

1. 写入 concept/topic 前尝试读取已有页面。
2. 若页面存在：
   - 合并 `sources`（按 `source_id` 去重）。
   - 合并正文中的相关判断条目（去重 + 限长）。
3. 若页面不存在：按现有模板创建。
4. source 页保持一文一页，不做跨文聚合。
5. 页面增长控制：
   - concept/topic 正文只保留最近与高价值条目窗口，防止无限膨胀。

预期结果：同一概念/主题页能随着新文写入而增强，不再被最后一次写入覆盖。

### 4.3 Index 分层导航

目标：把根 `index.md` 从日志式堆积改为可读总览。

策略：

1. 根 `index.md` 只保留：
   - 专家信息与 vault 总览。
   - 到 `sources/index.md`、`topics/index.md`、`concepts/index.md` 的入口链接。
2. 新增（或重建）分层索引：
   - `sources/index.md`: source 页面列表（可按标题或更新时间排序）。
   - `topics/index.md`: topic 页面列表（可附来源数）。
   - `concepts/index.md`: concept 页面列表（可附来源数）。
3. 每次 `build-wiki` 写入后，统一重建分层索引，避免增量追加带来的重复与膨胀。

预期结果：根 index 体积明显下降，导航体验可维护。

## 5. 影响文件（预计）

- [analyzer.py](/D:/Project/Expert_Digest/src/expert_digest/wiki/analyzer.py)
- [writer.py](/D:/Project/Expert_Digest/src/expert_digest/wiki/writer.py)
- [vault.py](/D:/Project/Expert_Digest/src/expert_digest/wiki/vault.py)
- [models.py](/D:/Project/Expert_Digest/src/expert_digest/wiki/models.py)（如需补充聚合辅助字段）
- 新增/更新测试：
  - `tests/test_wiki_analyzer.py`
  - `tests/test_wiki_writer.py`
  - `tests/test_wiki_vault.py`
  - 可能新增 `tests/test_wiki_indexes.py`

## 6. 验收标准

在真实 `zhihu_huang.sqlite3`（824 篇）上复跑后满足：

1. `source_page_count == 824`。
2. `traceability_ratio == 1.0` 且 `coverage_ratio == 1.0`。
3. concept/topic 数量较当前基线显著下降（以报告对比呈现）。
4. 根 `index.md` 由日志式追加改为总览入口，体积显著下降。
5. 抽检 concept/topic 页可见多来源聚合，不再被覆盖式写入破坏。
6. 现有 CLI 命令仍可用，`pytest` 与 `ruff check .` 通过。

## 7. 风险与回滚

### 7.1 风险

1. 过滤过严导致召回下降，漏掉有价值主题词。
2. 聚合策略不当导致页面正文重复或增长过快。
3. 索引重建策略与现有写入流程耦合，可能引入性能回归。

### 7.2 缓解

1. 通过测试覆盖典型输入，并保留可调阈值常量。
2. 先实现严格去重与长度限制，再观察真实数据结果。
3. 用真实数据跑一次端到端计时，记录与当前基线对比。

### 7.3 回滚

若质量回归或性能明显恶化，可回退到当前 MVP 分支基线，并保留本轮规则迭代为实验分支。

## 8. 实施顺序建议

1. analyzer 规则收敛与测试补全。
2. writer 聚合逻辑与页面去重策略。
3. index 分层重建与导航测试。
4. 真实 824 篇数据复跑，输出对比报告。
