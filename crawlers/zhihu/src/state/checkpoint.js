import fs from "node:fs/promises";

export async function loadSeenIdsFromJsonl(filePath) {
  const ids = new Set();
  try {
    const content = await fs.readFile(filePath, "utf8");
    for (const line of content.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed?.source_id != null) {
          ids.add(String(parsed.source_id));
        }
      } catch {
        // Ignore malformed lines and continue.
      }
    }
  } catch (error) {
    if (error?.code !== "ENOENT") {
      throw error;
    }
  }
  return ids;
}

export async function loadSeenIds(filePaths) {
  const all = new Set();
  for (const filePath of filePaths) {
    const ids = await loadSeenIdsFromJsonl(filePath);
    for (const id of ids) {
      all.add(id);
    }
  }
  return all;
}
