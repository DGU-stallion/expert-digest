# Xueqiu (雪球) Crawler

平台标识：`xueqiu`

## 占位说明

雪球爬虫待实现。实现后需满足：

1. 输出 `content_index.jsonl` 遵循 `crawlers/shared/schema.md` 规范
2. `platform` 字段值为 `"xueqiu"`
3. 默认输出目录：`../../data/crawlers/xueqiu`（相对于 `crawlers/xueqiu/`）
4. 在 `src/expert_digest/ingest/xueqiu_loader.py` 中实现 `XueqiuLoader`
5. 在 `src/expert_digest/ingest/loader.py` 中注册
