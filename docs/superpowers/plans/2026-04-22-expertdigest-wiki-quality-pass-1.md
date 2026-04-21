# ExpertDigest Wiki Quality Pass 1 实施计划

Date: 2026-04-22

Spec:

- `docs/superpowers/specs/2026-04-22-expertdigest-wiki-quality-pass-1-design.md`

## 目标

在保持 deterministic 的前提下，完成以下三项质量改进：

1. analyzer 过滤与归一化，减少低信息 concept/topic。
2. writer 聚合写入，concept/topic 支持多来源累积。
3. index 分层导航，根 `index.md` 变总览入口。

## 任务拆分

### Task 1: Analyzer 规则收敛

Files:

- `src/expert_digest/wiki/analyzer.py`
- `tests/test_wiki_analyzer.py`

Steps:

1. 添加噪声过滤测试（问句模板、日期数字碎片、低信息短词）。
2. 运行 `python -m pytest tests/test_wiki_analyzer.py -q`，确认失败。
3. 实现 deterministic 过滤规则（停用词扩展、噪声模式过滤、短词白名单）。
4. 运行 `python -m pytest tests/test_wiki_analyzer.py tests/test_evidence_builder.py -q`。
5. 提交：
   - `git add src/expert_digest/wiki/analyzer.py tests/test_wiki_analyzer.py`
   - `git commit -m "feat: refine wiki analyzer quality filters"`

### Task 2: Writer 聚合写入

Files:

- `src/expert_digest/wiki/writer.py`
- `tests/test_wiki_writer.py`

Steps:

1. 添加聚合测试：重复写入 concept/topic 时 source refs 累积且去重。
2. 运行 `python -m pytest tests/test_wiki_writer.py -q`，确认失败。
3. 实现页面读取-合并-回写逻辑；限制正文增长。
4. 运行 `python -m pytest tests/test_wiki_writer.py tests/test_wiki_vault.py -q`。
5. 提交：
   - `git add src/expert_digest/wiki/writer.py tests/test_wiki_writer.py`
   - `git commit -m "feat: aggregate wiki concept and topic pages"`

### Task 3: Index 分层导航

Files:

- `src/expert_digest/wiki/writer.py`（或辅助模块）
- `src/expert_digest/wiki/vault.py`（如需辅助方法）
- `tests/test_wiki_writer.py`
- `tests/test_wiki_vault.py`

Steps:

1. 添加测试：写入后生成 `sources/index.md`、`topics/index.md`、`concepts/index.md`。
2. 运行相关测试确认失败。
3. 实现根 index 总览 + 分层 index 重建。
4. 运行：
   - `python -m pytest tests/test_wiki_writer.py tests/test_wiki_vault.py -q`
   - `python -m pytest tests/test_cli_wiki.py -q`
5. 提交：
   - `git add src/expert_digest/wiki/writer.py src/expert_digest/wiki/vault.py tests/test_wiki_writer.py tests/test_wiki_vault.py`
   - `git commit -m "feat: add layered wiki indexes"`

### Task 4: 回归与真实数据验收

Steps:

1. 运行：
   - `python -m pytest -q`
   - `.\\.venv\\Scripts\\python.exe -m ruff check . --no-cache`
2. 真实数据复跑：
   - `build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild`
   - `build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang ...`
   - `search-wiki "泡泡玛特 核心能力" --wiki-root data/wiki/huang`
   - `eval-wiki --wiki-root data/wiki/huang --expected-source-count 824`
3. 更新验证报告（对比概念/主题数量、index 体积变化）。
4. 提交报告更新。

## 质量门

- 所有新增/修改测试通过。
- 全量 `pytest` 通过。
- `ruff check .` 通过。
- `eval-wiki` 维持 `traceability_ratio=1.0` 与 `coverage_ratio=1.0`。
- 实测 concept/topic 数量与根 index 体积相对基线下降。
