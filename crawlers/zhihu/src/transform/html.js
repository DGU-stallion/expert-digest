const ENTITY_MAP = {
  nbsp: " ",
  lt: "<",
  gt: ">",
  amp: "&",
  quot: '"',
  apos: "'"
};

function decodeEntities(input) {
  return String(input ?? "").replace(/&([a-z]+);/gi, (_m, key) => ENTITY_MAP[key.toLowerCase()] ?? _m);
}

function stripTags(input) {
  return String(input ?? "")
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<\/(p|div|h[1-6]|li|blockquote)>/gi, "\n\n")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, "");
}

function cleanupText(input) {
  return String(input ?? "")
    .replace(/\r/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function htmlToText(html) {
  return cleanupText(decodeEntities(stripTags(html)));
}

export function htmlToMarkdown(html) {
  let out = String(html ?? "");

  out = out.replace(/<script[\s\S]*?<\/script>/gi, "");
  out = out.replace(/<style[\s\S]*?<\/style>/gi, "");

  for (let level = 6; level >= 1; level -= 1) {
    const re = new RegExp(`<h${level}[^>]*>([\\s\\S]*?)<\\/h${level}>`, "gi");
    out = out.replace(re, (_m, content) => `${"#".repeat(level)} ${htmlToText(content)}\n\n`);
  }

  out = out.replace(/<a[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi, (_m, href, text) => {
    return `[${htmlToText(text)}](${href})`;
  });

  out = out.replace(/<(strong|b)[^>]*>([\s\S]*?)<\/\1>/gi, (_m, _tag, content) => `**${htmlToText(content)}**`);
  out = out.replace(/<(em|i)[^>]*>([\s\S]*?)<\/\1>/gi, (_m, _tag, content) => `*${htmlToText(content)}*`);
  out = out.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, (_m, content) => `- ${htmlToText(content)}\n`);
  out = out.replace(/<\/(p|div|blockquote)>/gi, "\n\n");
  out = out.replace(/<br\s*\/?>/gi, "\n");
  out = out.replace(/<[^>]+>/g, "");

  return cleanupText(decodeEntities(out));
}

export function extractImageUrls(html) {
  const source = String(html ?? "");
  const seen = new Set();
  const ordered = [];
  const regex = /<img[^>]+>/gi;
  let match = regex.exec(source);
  while (match) {
    const tag = match[0];
    const srcMatch =
      /data-original=["']([^"']+)["']/i.exec(tag) ??
      /data-actualsrc=["']([^"']+)["']/i.exec(tag) ??
      /src=["']([^"']+)["']/i.exec(tag);
    if (srcMatch?.[1]) {
      const url = srcMatch[1];
      if (!seen.has(url)) {
        seen.add(url);
        ordered.push(url);
      }
    }
    match = regex.exec(source);
  }
  return ordered;
}
