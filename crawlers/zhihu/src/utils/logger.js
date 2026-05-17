import fs from "node:fs/promises";
import path from "node:path";

function nowIso() {
  return new Date().toISOString();
}

export function createLogger(logFilePath) {
  const ready = logFilePath ? fs.mkdir(path.dirname(logFilePath), { recursive: true }) : Promise.resolve();

  async function emit(level, message, meta = null) {
    const payload = {
      time: nowIso(),
      level,
      message,
      meta
    };
    const text = `[${payload.time}] [${level.toUpperCase()}] ${message}`;
    if (level === "error") {
      console.error(text);
    } else if (level === "warn") {
      console.warn(text);
    } else {
      console.log(text);
    }
    if (logFilePath) {
      await ready;
      await fs.appendFile(logFilePath, `${JSON.stringify(payload)}\n`, "utf8");
    }
  }

  return {
    info(message, meta = null) {
      return emit("info", message, meta);
    },
    warn(message, meta = null) {
      return emit("warn", message, meta);
    },
    error(message, meta = null) {
      return emit("error", message, meta);
    }
  };
}
