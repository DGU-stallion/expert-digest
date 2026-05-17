import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { loadSeenIdsFromJsonl } from "../src/state/checkpoint.js";

test("loadSeenIdsFromJsonl reads existing source ids", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "zhihu-checkpoint-"));
  const file = path.join(dir, "content_index.jsonl");
  await fs.writeFile(
    file,
    [
      JSON.stringify({ source_id: "1" }),
      JSON.stringify({ source_id: "2" }),
      JSON.stringify({ source_id: "2" })
    ].join("\n"),
    "utf8"
  );

  const ids = await loadSeenIdsFromJsonl(file);
  assert.equal(ids.has("1"), true);
  assert.equal(ids.has("2"), true);
  assert.equal(ids.size, 2);
});
