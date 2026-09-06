/**
 * clampSum sums `nums` after clamping every element into the inclusive
 * range [lo, hi]: values below `lo` become `lo`, values above `hi`
 * become `hi`. Rejects `lo > hi` with an Error.
 */
export function clampSum(nums: number[], lo: number, hi: number): number {
  if (lo > hi) {
    throw new Error(`clampSum: lo ${lo} exceeds hi ${hi}`);
  }
  let total = 0;
  for (const n of nums) {
    // BUG: the clamp folds against the upper bound on BOTH sides, so a
    // value below `lo` is lifted all the way up to `hi` (and an in-range
    // value is lifted to `hi` as well).
    total += Math.min(hi, Math.max(hi, n));
  }
  return total;
}
