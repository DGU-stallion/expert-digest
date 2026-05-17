export function sleep(ms) {
  const waitMs = Math.max(0, Number(ms) || 0);
  return new Promise(resolve => setTimeout(resolve, waitMs));
}

export function randomInt(min, max) {
  const a = Number(min) || 0;
  const b = Number(max) || 0;
  const low = Math.min(a, b);
  const high = Math.max(a, b);
  return Math.floor(Math.random() * (high - low + 1)) + low;
}

export async function delayWithJitter(baseMs, jitterMinMs, jitterMaxMs) {
  const total = Math.max(0, Number(baseMs) || 0) + randomInt(jitterMinMs, jitterMaxMs);
  await sleep(total);
  return total;
}
