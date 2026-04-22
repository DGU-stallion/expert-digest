# ExpertDigest Handbook/Skill LLM-Only 重构实施计划

Date: 2026-04-22

Spec:

- `docs/superpowers/specs/2026-04-22-expertdigest-handbook-skill-llm-book-design.md`

> 执行要求：按 `superpowers:subagent-driven-development` 流程逐任务推进。每个任务必须遵循：
> 1) 先写失败测试；
> 2) 运行确认失败；
> 3) 最小实现；
> 4) 运行相关测试；
> 5) 提交 commit；
> 6) 主 agent review 通过后进入下一任务。

## 目标

在不扩展到 UI、多 Agent、向量库绑定的前提下，完成 Handbook/Skill 产物重构：

1. handbook 从“检索报告”切换为“学习型书籍”结构。
2. 正文不出现文章链接与文章推荐列表。
3. 用社区发现替换当前聚类主路径，提升主题成章质量。
4. 删除 deterministic fallback，内容生成阶段 LLM 必须参与。
5. 保留内部 trace sidecar 用于可追溯与质检。

## 范围与边界

### 在范围内

- `generate-handbook` / `generate-skill-draft` 生成链路重构。
- topic graph + community detection。
- LLM required 错误语义。
- handbook/skill 新契约测试与质量门扩展。

### 不在范围内

- Streamlit 重做。
- 多 Agent 运行时。
- deep answer 全能力。
- 引入强依赖向量数据库。

## 任务拆分

### Task 1: LLM Required 契约与 CLI fail-fast

Files:

- `src/expert_digest/cli.py`
- `src/expert_digest/generation/llm_client.py`（如需补充 preflight 辅助）
- `tests/test_cli.py`
- `tests/test_cli_author_profile.py`（如涉及 skill 命令路径）

Steps:

1. 新增失败测试：
   - 当 `generate-handbook --synthesis-mode hybrid` 无可用 LLM client 时返回非 0。
   - 当 `generate-skill-draft` 无可用 LLM client 时返回非 0。
   - 错误消息包含标准错误码（如 `llm_client_unavailable`）。
2. 运行：
   - `python -m pytest tests/test_cli.py tests/test_cli_author_profile.py -q`
   - 预期失败。
3. 最小实现：
   - 移除内容生成路径中的 deterministic fallback。
   - 在命令入口增加 preflight 检查，LLM 不可用直接失败退出。
4. 运行相关测试直至通过。
5. 提交：
   - `git add src/expert_digest/cli.py src/expert_digest/generation/llm_client.py tests/test_cli.py tests/test_cli_author_profile.py`
   - `git commit -m "feat: require llm for handbook and skill generation"`

### Task 2: Handbook 书籍型输出契约重构

Files:

- `src/expert_digest/generation/handbook_writer.py`
- `src/expert_digest/domain/models.py`（如需扩展 handbook 结构）
- `tests/test_handbook_writer.py`（若不存在则新增）
- `tests/test_cli.py`

Steps:

1. 新增失败测试：
   - 输出包含：作者简介、引言、目录、第 1 章...第 N 章、结语。
   - 正文不包含 URL 模式（`http://` / `https://`）与“推荐文章”区块。
2. 运行对应测试确认失败。
3. 最小实现：
   - 重写 handbook 渲染模板为书籍结构。
   - 将来源信息迁移到内部 sidecar，不写入 reader-facing markdown。
4. 运行：
   - `python -m pytest tests/test_handbook_writer.py tests/test_cli.py -q`
5. 提交：
   - `git add src/expert_digest/generation/handbook_writer.py src/expert_digest/domain/models.py tests/test_handbook_writer.py tests/test_cli.py`
   - `git commit -m "feat: switch handbook output to book-style format"`

### Task 3: Topic Graph 与 Community Detection 主路径

Files:

- 新增 `src/expert_digest/knowledge/topic_graph.py`
- 新增 `src/expert_digest/knowledge/community_detection.py`
- `src/expert_digest/knowledge/topic_clusterer.py`
- `tests/test_topic_graph.py`
- `tests/test_topic_clusterer.py`

Steps:

1. 新增失败测试：
   - 给定样本语料可形成稳定社区划分。
   - 小规模噪声社区被过滤或合并。
2. 运行：
   - `python -m pytest tests/test_topic_graph.py tests/test_topic_clusterer.py -q`
   - 预期失败。
3. 最小实现：
   - 构建 topic graph（语义相似 + 共现信号）。
   - 引入社区发现并接入现有 topic 输出接口。
   - 保留旧聚类实现仅供调试，不作为主路径。
4. 运行相关测试通过。
5. 提交：
   - `git add src/expert_digest/knowledge/topic_graph.py src/expert_digest/knowledge/community_detection.py src/expert_digest/knowledge/topic_clusterer.py tests/test_topic_graph.py tests/test_topic_clusterer.py`
   - `git commit -m "feat: add community detection for chapter topic discovery"`

### Task 4: 分阶段 LLM 写作链路（章节规划 -> 分章写作 -> 全书润色）

Files:

- 新增 `src/expert_digest/generation/book_pipeline.py`
- `src/expert_digest/generation/handbook_writer.py`
- `src/expert_digest/generation/llm_client.py`
- `tests/test_book_pipeline.py`
- `tests/test_handbook_writer.py`

Steps:

1. 新增失败测试：
   - chapter plan 失败时返回 `chapter_plan_failed`。
   - chapter drafting 失败时返回 `chapter_draft_failed`。
   - 全书产物章节间术语字段一致（最小一致性断言）。
2. 运行测试确认失败。
3. 最小实现：
   - 新增 book pipeline 三阶段执行器。
   - 将 handbook 生成切换为该 pipeline。
4. 运行：
   - `python -m pytest tests/test_book_pipeline.py tests/test_handbook_writer.py tests/test_cli.py -q`
5. 提交：
   - `git add src/expert_digest/generation/book_pipeline.py src/expert_digest/generation/handbook_writer.py src/expert_digest/generation/llm_client.py tests/test_book_pipeline.py tests/test_handbook_writer.py tests/test_cli.py`
   - `git commit -m "feat: add llm-driven multi-stage book generation pipeline"`

### Task 5: Skill 生成重构为 LLM-only 结构化输出

Files:

- `src/expert_digest/knowledge/skill_writer.py`
- `src/expert_digest/cli.py`
- `tests/test_skill_writer.py`
- `tests/test_cli_author_profile.py`

Steps:

1. 新增失败测试：
   - skill 输出包含：风格原则、回答流程、风险守则、拒答策略。
   - 无 LLM client 时命令失败并输出标准错误码。
2. 运行测试确认失败。
3. 最小实现：
   - skill 文本由 LLM 生成并按固定结构落盘。
4. 运行：
   - `python -m pytest tests/test_skill_writer.py tests/test_cli_author_profile.py -q`
5. 提交：
   - `git add src/expert_digest/knowledge/skill_writer.py src/expert_digest/cli.py tests/test_skill_writer.py tests/test_cli_author_profile.py`
   - `git commit -m "feat: generate structured skill draft with llm only"`

### Task 6: Trace Sidecar 与质量门扩展

Files:

- `src/expert_digest/generation/handbook_writer.py`
- `src/expert_digest/wiki/evaluator.py`（如新增书籍质量指标）
- 新增 `tests/test_handbook_quality.py`
- `tests/test_cli.py`

Steps:

1. 新增失败测试：
   - handbook sidecar 存在并覆盖正文段落映射。
   - 质量门可检查“结构完整、正文无链接、重复率阈值”。
2. 运行测试确认失败。
3. 最小实现：
   - 写出 trace sidecar（internal JSON）。
   - 增加书籍质量判定逻辑并接入 CLI。
4. 运行：
   - `python -m pytest tests/test_handbook_quality.py tests/test_cli.py -q`
5. 提交：
   - `git add src/expert_digest/generation/handbook_writer.py src/expert_digest/wiki/evaluator.py tests/test_handbook_quality.py tests/test_cli.py`
   - `git commit -m "feat: add handbook trace sidecar and quality checks"`

### Task 7: 全量回归与真实数据验收

Steps:

1. 运行全量验证：
   - `python -m pytest -q`
   - `python -m ruff check .`
2. 真实数据跑通：
   - `build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild`
   - `build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang_ready ...`
   - `generate-handbook ... --synthesis-mode hybrid --wiki-root-for-quality data/wiki/huang_ready ...`
   - `generate-skill-draft ... --wiki-root-for-quality data/wiki/huang_ready ...`
3. 验收检查：
   - handbook 正文无链接；
   - 结构完整；
   - LLM 未配置场景可稳定失败；
   - 质量报告达标并输出最终验证文档。
4. 提交验证文档与必要脚本更新。

## 质量门

1. 新增/修改测试全部通过。
2. 全量 `pytest` 通过。
3. `ruff check .` 通过。
4. 真实数据产物满足“书籍结构 + 正文零链接 + LLM required”。
5. 质量报告与 trace sidecar 可复核。

## 风险控制

1. 若社区发现参数导致章节碎片化，先回退参数，不回退主路径设计。
2. 若 LLM 稳定性不足，优先完善错误语义和重试策略，不恢复 deterministic fallback。
3. 若生成时延过高，后续迭代引入分章并发与缓存，不在本计划扩大范围。
