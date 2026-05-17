import fs from "node:fs/promises";
import path from "node:path";

async function ensureParentDir(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

export class JsonlWriter {
  constructor(filePath) {
    this.filePath = filePath;
    this.ready = ensureParentDir(filePath);
  }

  async append(record) {
    await this.ready;
    const line = `${JSON.stringify(record)}\n`;
    await fs.appendFile(this.filePath, line, "utf8");
  }
}

export async function writeJson(filePath, value) {
  await ensureParentDir(filePath);
  await fs.writeFile(filePath, JSON.stringify(value, null, 2), "utf8");
}
