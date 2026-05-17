/**
 * Agent-driven data ingestion for Zhihu crawler.
 *
 * Accepts raw Zhihu API response data (fetched by agent via MCP Playwright tools),
 * normalizes it, and writes to the standard crawler output directory.
 *
 * Usage:
 *   node src/agent-crawl.js \
 *     --user-token huang-wei-yan-30 \
 *     --output-dir data/crawlers/zhihu \
 *     --profile-file /tmp/profile.json \
 *     --answers-file /tmp/answers.json \
 *     --articles-file /tmp/articles.json
 */

import fs from "node:fs/promises";
import path from "node:path";

import { normalizeAnswer, normalizeArticle } from "./normalize/normalizer.js";
import { createLogger } from "./utils/logger.js";
import { JsonlWriter, writeJson } from "./writer/jsonlWriter.js";

function stripHtml(html) {
  return (html || "")
    .replace(/<[^>]*>/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

async function readJsonFile(filePath) {
  const text = await fs.readFile(filePath, "utf-8");
  return JSON.parse(text);
}

export async function agentCrawl(config) {
  const { userToken, outputDir, profileFile, answersFile, articlesFile, markdownEnabled } = config;

  const root = path.join(outputDir, userToken);
  const rawDir = path.join(root, "raw");
  const indexDir = path.join(root, "index");
  const logsDir = path.join(root, "logs");
  await fs.mkdir(rawDir, { recursive: true });
  await fs.mkdir(indexDir, { recursive: true });
  await fs.mkdir(logsDir, { recursive: true });

  const logger = createLogger(path.join(logsDir, "crawl.log"));
  const crawlTime = new Date().toISOString();

  // --- Read input files ---
  const profile = await readJsonFile(profileFile);
  const answers = Array.isArray(await readJsonFile(answersFile)) ? await readJsonFile(answersFile) : [];
  const articles = Array.isArray(await readJsonFile(articlesFile)) ? await readJsonFile(articlesFile) : [];

  await logger.info("Agent crawl started", {
    userToken,
    profileName: profile?.name ?? "",
    answersCount: answers.length,
    articlesCount: articles.length,
  });

  // --- Save raw profile ---
  await writeJson(path.join(rawDir, "profile.json"), {
    fetched_at: crawlTime,
    user_token: userToken,
    profile,
  });

  // --- Save raw answers + normalized index ---
  const rawAnswersWriter = new JsonlWriter(path.join(rawDir, "answers.jsonl"));
  const rawArticlesWriter = new JsonlWriter(path.join(rawDir, "articles.jsonl"));
  const indexWriter = new JsonlWriter(path.join(indexDir, "content_index.jsonl"));

  for (const answer of answers) {
    const rowTime = new Date().toISOString();
    await rawAnswersWriter.append({
      source_type: "answer",
      source_id: String(answer.id ?? ""),
      user_token: userToken,
      fetched_at: rowTime,
      request: { method: "agent_mcp" },
      raw: answer,
    });

    const normalized = normalizeAnswer(answer, {
      userToken,
      crawlTime: rowTime,
      markdownEnabled: Boolean(markdownEnabled),
    });

    // Add top-level platform and content_text for loader dispatch
    await indexWriter.append({
      ...normalized,
      platform: "zhihu",
      content_text: stripHtml(answer.content ?? ""),
    });
  }

  // --- Save raw articles + normalized index ---
  for (const article of articles) {
    const rowTime = new Date().toISOString();
    await rawArticlesWriter.append({
      source_type: "article",
      source_id: String(article.id ?? ""),
      user_token: userToken,
      fetched_at: rowTime,
      request: { method: "agent_mcp" },
      raw: article,
    });

    const normalized = normalizeArticle(article, {
      userToken,
      crawlTime: rowTime,
      markdownEnabled: Boolean(markdownEnabled),
    });

    await indexWriter.append({
      ...normalized,
      platform: "zhihu",
      content_text: stripHtml(article.content ?? ""),
    });
  }

  await logger.info("Agent crawl completed", {
    answersWritten: answers.length,
    articlesWritten: articles.length,
    outputRoot: root,
  });

  return {
    userToken,
    outputRoot: root,
    totalItems: answers.length + articles.length,
  };
}

// --- CLI ---
function parseArgs(argv) {
  const config = {
    userToken: "",
    outputDir: "data/crawlers/zhihu",
    profileFile: "",
    answersFile: "",
    articlesFile: "",
    markdownEnabled: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case "--help":
      case "-h":
        config.help = true;
        break;
      case "--user-token":
        config.userToken = argv[++i] ?? "";
        break;
      case "--output-dir":
        config.outputDir = argv[++i] ?? "data/crawlers/zhihu";
        break;
      case "--profile-file":
        config.profileFile = argv[++i] ?? "";
        break;
      case "--answers-file":
        config.answersFile = argv[++i] ?? "";
        break;
      case "--articles-file":
        config.articlesFile = argv[++i] ?? "";
        break;
      case "--markdown":
        config.markdownEnabled = true;
        break;
    }
  }
  return config;
}

function helpText() {
  return `Zhihu agent-driven data ingestion

Usage:
  node src/agent-crawl.js [options]

Options:
  --user-token <token>       Zhihu user token (required)
  --output-dir <dir>         Output root directory (default: data/crawlers/zhihu)
  --profile-file <path>      Path to profile JSON file (required)
  --answers-file <path>      Path to answers JSON array file (required)
  --articles-file <path>     Path to articles JSON array file (required)
  --markdown                 Enable markdown content generation
  --help, -h                 Show this help
`;
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  if (config.help) {
    console.log(helpText());
    process.exit(0);
  }

  if (!config.userToken || !config.profileFile || !config.answersFile || !config.articlesFile) {
    console.error("Error: --user-token, --profile-file, --answers-file, --articles-file are required");
    console.error(helpText());
    process.exit(1);
  }

  try {
    const result = await agentCrawl(config);
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error("Agent crawl failed:");
    console.error(error?.stack ?? String(error));
    process.exit(1);
  }
}

main();
