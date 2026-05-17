/**
 * Zhihu crawler entry.
 *
 * Agent-driven data ingestion (agentCrawl) is the only supported flow.
 * The HTTP + Cookie approach (zhihuHttpFetcher) has been removed — Zhihu
 * consistently returns 40362 for pure HTTP requests. Data must be fetched
 * via MCP Playwright browser tools using the user's authenticated Chrome session.
 */

export { agentCrawl } from "./agent-crawl.js";
