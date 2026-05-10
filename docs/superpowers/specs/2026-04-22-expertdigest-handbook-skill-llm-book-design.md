# ExpertDigest Handbook/Skill LLM-Only 重构设计

Date: 2026-04-22

## 1. 背景与问题定义

当前 `handbook` 产物仍偏“检索报告”范式，主要问题：

1. 内容结构以“主题+文章池+观点摘录”为主，不是学习型书籍。
2. 正文出现大量文章链接，读者体验是资料列表而非系统化学习文本。
3. 聚类结果主题边界不稳定，章节可读性与完整性不足。
4. 生成链路允许 deterministic fallback，导致 LLM 未参与时产物质量显著下降且难以及时暴露问题。

本次重构目标是把输出从“证据展示文档”升级为“可学习、可连读、可追溯（内部）”的书籍型知识产品。

## 2. 目标与非目标

### 2.1 目标

1. **Handbook 形态重构**：输出固定为书籍结构：
   - 作者简介
   - 引言
   - 目录
   - 第 1 章 ... 第 N 章
   - 结语
2. **正文零链接**：正文不展示原始文章 URL/标题列表，不出现“推荐文章池”样式内容。
3. **主题成章能力升级**：用社区发现替换当前聚类主路径，提高章节边界质量。
4. **LLM 强制参与**：删除 deterministic fallback；内容生成阶段无 LLM 即失败。
5. **内部可追溯保留**：保留段落到 source/evidence 的内部 trace 文件，用于质检与审计。

### 2.2 非目标

1. 不改造 Streamlit UI。
2. 不引入多 Agent 架构。
3. 不引入向量数据库作为强依赖（Qdrant/Chroma/pgvector 等仍非必需）。
4. 不在本轮实现 deep answer 全能力。

## 3. 核心设计决策

### 3.1 产物契约切换：Report -> Book

新增 handbook 产物契约：

1. 面向读者只输出“学习文本”。
2. 引用关系仅存在于 sidecar trace（JSON），不进入正文。
3. 无法稳定归类或低置信内容进入“未采用素材池”（内部），不污染章节正文。

### 3.2 主题发现切换：聚类 -> 社区发现

在 Wiki 证据层之上构建主题图：

- 节点：候选知识单元（section/chunk 或其聚合表达）。
- 边：语义相似度 + 共现关系（可加结构邻近信号）。
- 社区发现：产出章节候选簇，过滤小簇与噪声簇后进入章节规划。

默认采用“图社区发现优先、旧聚类兜底仅用于调试”的策略；正式产物路径不再依赖旧聚类结果。

### 3.3 生成策略切换：LLM-Optional -> LLM-Required

以下步骤必须有可用 LLM：

1. 全书章节规划（chapter plan）
2. 分章正文生成
3. 全书一致性润色
4. skill 文本生成

若 LLM client 不可用、调用失败或鉴权失败，命令直接返回非 0 并输出明确错误原因（如 `llm_client_unavailable` / `llm_auth_failed` / `llm_timeout`）。

## 4. 新生成流水线

```text
Wiki Evidence
  -> Candidate Unit Builder
  -> Topic Graph Builder
  -> Community Detection
  -> Chapter Candidate Filter
  -> LLM Chapter Plan
  -> LLM Chapter Drafting
  -> LLM Global Coherence Pass
  -> Book Output (reader-facing markdown)
  -> Trace Sidecar (internal JSON)
```

### 4.1 Candidate Unit Builder

从 `ParentSection / ChildChunk / EvidenceSpan` 构造候选知识单元，附带：

- source_id 列表
- evidence span 引用
- 去噪后的主题短语

### 4.2 Topic Graph Builder + Community Detection

构造图后运行社区发现得到章节候选；过滤规则包括：

- 最小社区规模阈值
- 低信息社区阈值
- 高重复社区合并

### 4.3 Chapter Plan (LLM)

LLM 输出章节蓝图：

- 章节名
- 学习目标
- 关键概念与逻辑顺序
- 与前后章衔接关系

### 4.4 Chapter Drafting (LLM)

每章采用统一写作模板（对读者可见）：

1. 本章目标
2. 核心概念解释
3. 关键机制/框架
4. 案例与边界条件
5. 常见误区
6. 小结与行动建议

### 4.5 Global Coherence Pass (LLM)

对全书进行跨章一致性处理：

- 术语统一
- 重复段落压缩
- 逻辑断裂修补
- 语气和叙事节奏统一

## 5. Skill 产物重构

`generate-skill-draft` 同样改为 LLM 强制参与，目标从“标签摘要”升级为：

1. 风格原则（语言与推理偏好）
2. 回答流程（结论-证据-边界-行动）
3. 风险守则（高风险场景限制）
4. 拒答/降级策略

同样不允许 deterministic fallback。

## 6. CLI 与错误语义

### 6.1 CLI 行为

1. `generate-handbook` 默认走书籍型输出模板。
2. `generate-skill-draft` 默认走 LLM 生成。
3. 保留质量门参数（wiki eval/lint）作为生成前 gate。

### 6.2 错误语义

统一错误码/错误消息（至少）：

- `llm_client_unavailable`
- `llm_auth_failed`
- `llm_request_timeout`
- `llm_response_invalid`
- `chapter_plan_failed`
- `chapter_draft_failed`

## 7. 开源复用策略

原则：**核心域模型与产物契约自研，算法与工具层可复用**。

可复用范围：

1. 社区发现算法实现（Leiden/Louvain/Label Propagation 的成熟实现）
2. 图构建基础设施
3. 文本去重/相似度工具

不可外包范围：

1. `SourceDocument -> Evidence -> Wiki -> Book` 主流程控制权
2. Trace sidecar 数据契约
3. 质量门判定逻辑

## 8. 验收标准

在真实数据（当前 824 篇）上满足：

1. 生成结果包含固定书籍结构（作者简介/引言/目录/章节/结语）。
2. 正文不含外链 URL，不含“推荐文章列表”区块。
3. 章节主题可解释，低信息噪声章节显著减少。
4. LLM 未配置时生成命令失败（非 0），不再静默回退。
5. trace sidecar 覆盖正文段落（覆盖率阈值在实施计划中定义）。

## 9. 风险与缓解

1. **风险：** 社区发现参数敏感导致章节粒度波动。  
   **缓解：** 固化参数基线 + 回归集对比。
2. **风险：** LLM-only 导致可用性下降（配置缺失即失败）。  
   **缓解：** 提供明确 preflight 检查与错误提示。
3. **风险：** 长文本生成成本与时延上升。  
   **缓解：** 分章并发生成（后续迭代）+ 章节缓存。

## 10. 任务拆分清单（实施前）

### Task A: 产物契约重构
- 定义书籍型 handbook markdown schema（reader-facing）。
- 定义 trace sidecar schema（internal）。
- 更新验收样例与快照测试基线。

### Task B: 主题发现重构
- 增加 topic graph 构建器与社区发现模块。
- 加入社区过滤与合并策略。
- 建立与现有聚类结果的对比评估脚本。

### Task C: LLM 生成链路重构
- 章节规划器（LLM）
- 章节写作器（LLM）
- 全书一致性润色器（LLM）
- 删除 deterministic fallback，补齐错误语义。

### Task D: Skill 生成重构
- 以 LLM 驱动输出新版 skill 文本结构。
- 与 author profile、wiki quality gate 联动。

### Task E: 质量门与验收
- 扩展 `eval-wiki`/`lint-wiki` 到“书籍可读性”相关指标。
- 新增 handbook 结构/链接/重复率测试。
- 真实数据回归与质量报告。

---

本设计文档用于确认“方向与验收口径”；实现细节（测试先行顺序、文件级任务、提交粒度）在后续实施计划文档中细化。
