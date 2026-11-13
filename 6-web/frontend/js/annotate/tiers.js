// 草稿置信度分档：阈值、颜色、文案、计数。
//
// 为什么单独一个模块：分档规则同时被"画布上色"和"列表项样式"用到，
// 两边一旦各写一套阈值，同一批草稿会出现两种颜色，看着像 bug。
// 这里只放纯数据与纯函数，不碰 DOM、不读全局 state，方便直接跑 node 单测。
//
// 单测：node 6-web/frontend/tests/test_tiers.mjs

/**
 * 高置信度阈值。
 *
 * `0.70` 是 Ultralytics 默认 val 的置信度基准之上，现场审核时"基本不用看"的那一档；
 * `0.40` 以下基本是模型在猜，需要人逐条看。两侧都取闭区间上界，
 * 避免 0.7 / 0.4 正好落在缝里（0.7 归 high、0.4 归 mid）。
 */
export const TIER_HIGH = 0.7;
export const TIER_MID = 0.4;

/** 画布填充色（RGB 数组）。与 style.css 的 --ok / --amber / --danger 取同一组值。 */
export const TIER_COLORS = Object.freeze({
  high: [52, 211, 153],
  mid: [251, 191, 36],
  low: [248, 113, 113],
  // unknown 用琥珀色而不是红色：通道没给置信度不等于"模型在猜"，
  // 整批标红会让人不敢信绿框。
  unknown: [251, 191, 36],
});

/** 徽标文案。unknown 用问号，避免和"低"混淆。 */
export const TIER_TEXT = Object.freeze({ high: "高", mid: "中", low: "低", unknown: "?" });

/** 档位名的固定顺序，计数结果的键顺序以此为准。 */
export const TIER_ORDER = Object.freeze(["high", "mid", "low", "unknown"]);

/**
 * 置信度 → 档位。
 *
 * - `score < 0`、`null`、`""` 或非有限值：`unknown`（"通道没给置信度"，外部服务常见）；
 *   `null` 单独判一次，因为 `Number(null) === 0` 会把它误算成"低置信度"，
 *   整批红框看起来像模型在猜；
 * - 其余按阈值落 high / mid / low，边界值归上一档（`>=`）。
 */
export function tierOf(score) {
  if (score === null || score === "") return "unknown";
  const value = Number(score);
  if (!Number.isFinite(value) || value < 0) return "unknown";
  if (value >= TIER_HIGH) return "high";
  if (value >= TIER_MID) return "mid";
  return "low";
}

/** 档位 → 画布 RGB 数组（调用方自行拼 `rgb()`）。 */
export function tierColor(score) {
  return TIER_COLORS[tierOf(score)];
}

/** 档位 → 徽标中文。 */
export function tierText(score) {
  return TIER_TEXT[tierOf(score)];
}

/** 档位 → CSS 类名（`tier-high` 等），画布与列表共用同一套命名。 */
export function tierClass(score) {
  return `tier-${tierOf(score)}`;
}

/**
 * 按档位统计草稿数量。
 *
 * 入参可以是数组，也可以是任何可迭代对象（例如 `Map.values()`），
 * 元素只要有 `score` 字段即可；缺失 score 的元素按 unknown 计。
 * 返回 `{ high, mid, low, unknown, total }`，键顺序固定，便于对比断言。
 */
export function countByTier(drafts) {
  const counts = { high: 0, mid: 0, low: 0, unknown: 0, total: 0 };
  for (const draft of drafts || []) {
    const tier = tierOf(draft ? draft.score : undefined);
    counts[tier] += 1;
    counts.total += 1;
  }
  return counts;
}

/** 计数结果 → 一行中文摘要（给 tooltip 用，空集合返回空串）。 */
export function tierCountText(counts) {
  if (!counts || !counts.total) return "";
  return TIER_ORDER.filter((tier) => counts[tier] > 0)
    .map((tier) => `${TIER_TEXT[tier]} ${counts[tier]}`)
    .join(" / ");
}
