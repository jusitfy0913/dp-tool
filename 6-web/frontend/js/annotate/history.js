// 撤销/重做双栈：纯函数版，不持有任何全局状态、不碰 DOM。
//
// 调用方（annotate.js）把两个栈当成不可变数组传进来，拿返回的新数组去替换：
//   state.undo = result.undo;
// 这样测试里可以随便往返几十次而不用先造一个页面。
//
// 语义与 Qt 版一致：undo 栈存"改动前"的快照；每次改动前先 pushSnapshot，
// 且 push 会清空 redo（分叉的历史没有意义）。
//
// 单测：node 6-web/frontend/tests/test_history.mjs

/** 撤销历史深度。对齐 Qt 版的 50，兼顾内存与"够用"。 */
export const HISTORY_LIMIT = 50;

/**
 * 压入一个快照（改动**之前**调用）。
 *
 * 返回 `{ undo, redo, dropped }`：
 * - `undo` 新栈（已按深度上限从头裁掉多余项）；
 * - `redo` 恒为空数组——新改动让旧的重做分支失效；
 * - `dropped` 本次因超深被丢弃的条数（0 或 1，上限为 0 时可能更大），供测试与埋点核对。
 *
 * `entry` 为 null（例如没有打开的工程）时原样返回，不做任何改动。
 */
export function pushSnapshot({ undo = [], entry = null, limit = HISTORY_LIMIT } = {}) {
  if (!entry) return { undo: [...undo], redo: [], dropped: 0 };
  const next = [...undo, entry];
  const depth = Math.max(0, Math.trunc(limit));
  const dropped = Math.max(0, next.length - depth);
  return { undo: dropped ? next.slice(dropped) : next, redo: [], dropped };
}

/**
 * 撤销一步。
 *
 * `current` 是"撤销前"的快照（通常是当前状态），会被压进 redo 栈，
 * 这样重做才能精确回到按 Ctrl+Z 之前的那一帧。
 *
 * 返回 `{ ok, entry, undo, redo }`：栈空时 `ok=false`，其余字段为原栈副本。
 */
export function undoStep({ undo = [], redo = [], current = null } = {}) {
  if (!undo.length) return { ok: false, entry: null, undo: [...undo], redo: [...redo] };
  const nextUndo = undo.slice(0, -1);
  const entry = undo[undo.length - 1];
  const nextRedo = current ? [...redo, current] : [...redo];
  return { ok: true, entry, undo: nextUndo, redo: nextRedo };
}

/** 重做一步：与 `undoStep` 对称，`current` 会回到 undo 栈。 */
export function redoStep({ undo = [], redo = [], current = null } = {}) {
  if (!redo.length) return { ok: false, entry: null, undo: [...undo], redo: [...redo] };
  const nextRedo = redo.slice(0, -1);
  const entry = redo[redo.length - 1];
  const nextUndo = current ? [...undo, current] : [...undo];
  return { ok: true, entry, undo: nextUndo, redo: nextRedo };
}

/** 还能不能撤销。 */
export function canUndo(undo = []) {
  return undo.length > 0;
}

/** 还能不能重做。 */
export function canRedo(redo = []) {
  return redo.length > 0;
}

/** 两个栈 → 按钮状态（`disabled` 取反），给 `updateHistoryButtons` 直接用。 */
export function historyButtons(undo = [], redo = []) {
  return { undoDisabled: !canUndo(undo), redoDisabled: !canRedo(redo) };
}
