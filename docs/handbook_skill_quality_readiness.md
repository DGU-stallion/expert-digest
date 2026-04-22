# Handbook/Skill 高质量生成前准备清单

Date: 2026-04-22

## 目标

在正式生成高质量 `handbook` / `skill` 前，先完成可重复的质量准备与门禁：

1. Wiki 质量门：`eval-wiki` + `lint-wiki`。
2. 生成前门禁：`generate-handbook` / `generate-skill-draft` 支持 quality gate 参数。
3. 数据准备完整：`build-evidence` + `rebuild-embeddings` 已完成。
4. 作者画像主题提炼降噪：优先输出稳定领域主题而非问题句碎片。

## 已完成改动

### 1) Analyzer 噪声收敛

- 修复两字母小写噪声缩写泄漏（如 `bd` / `bp`）。
- 保留白名单短词（如 `AI`、`IP`），并限制短缩写放行范围。

影响文件：

- `src/expert_digest/wiki/analyzer.py`
- `tests/test_wiki_analyzer.py`

### 2) Author Profile 主题提炼升级

- `focus_topics` 优先采用规则化主题映射（宏观/交易/地产/行业/方法论）。
- 对问句模板、数字碎片、低信息标题片段做过滤。

影响文件：

- `src/expert_digest/knowledge/author_profile.py`
- `tests/test_author_profile.py`

### 3) 生成前质量门（CLI）

`generate-handbook` 与 `generate-skill-draft` 新增可选参数：

- `--wiki-root-for-quality`
- `--expected-source-count-for-quality`
- `--max-lint-issues-for-quality`

当质量门不通过时，命令失败并返回非 0：

- `traceability_ratio < 1.0` 拒绝
- `coverage_ratio < 1.0`（若提供 expected_source_count）拒绝
- `issue_count > max_lint_issues` 拒绝

影响文件：

- `src/expert_digest/cli.py`
- `tests/test_cli.py`
- `tests/test_cli_author_profile.py`

## 准备数据与验收结果

### 数据准备命令

```powershell
.\.venv\Scripts\expert-digest.exe build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild
.\.venv\Scripts\expert-digest.exe rebuild-embeddings --db data/processed/zhihu_huang.sqlite3 --model hash-bow-v1 --dim 256
.\.venv\Scripts\expert-digest.exe build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang_ready --expert-id huang --expert-name "黄彦臻" --purpose "沉淀黄彦臻公开文章中的投资分析框架。"
```

### Wiki 门禁结果（`data/wiki/huang_ready`）

- `eval-wiki`: `traceability_ratio=1.0`，`coverage_ratio=1.0`
- `lint-wiki`: `issue_count=2`

输出文件：

- `data/outputs/wiki_eval_huang_ready.json`
- `data/outputs/wiki_lint_huang_ready.json`

### 质量门 smoke test（已通过）

```powershell
.\.venv\Scripts\expert-digest.exe generate-handbook --db data/processed/zhihu_huang.sqlite3 --author 黄彦臻 --output data/outputs/huang_handbook_ready.md --synthesis-mode deterministic --theme-source cluster --max-themes 5 --num-topics 8 --wiki-root-for-quality data/wiki/huang_ready --expected-source-count-for-quality 824 --max-lint-issues-for-quality 10

.\.venv\Scripts\expert-digest.exe generate-skill-draft --db data/processed/zhihu_huang.sqlite3 --author 黄彦臻 --output data/outputs/huang_skill_ready.md --wiki-root-for-quality data/wiki/huang_ready --expected-source-count-for-quality 824 --max-lint-issues-for-quality 10
```

## 下一步（正式高质量生成）

在当前准备完成后，可直接进入最终生成：

1. Handbook：优先使用 `--synthesis-mode hybrid`，并固定 quality gate 参数。
2. Skill：使用 quality gate + 最新 `build-author-profile` 输出。
3. 若需更高文字质量，可在保持质量门通过的前提下做一次手工抽检并微调 `max-themes / num-topics`。
