import { test } from "node:test";
import assert from "node:assert/strict";

import { clampSum } from "./clamp.ts";

test("sums in-range values unchanged", () => {
  assert.equal(clampSum([1, 2, 3], 0, 5), 6);
});

test("values below lo are clamped up to lo, not hi", () => {
  assert.equal(clampSum([-5, 0, 5], 0, 10), 5);
});

test("values above hi are clamped down to hi", () => {
  assert.equal(clampSum([10, 20], 0, 5), 10);
});

test("degenerate lo === hi range", () => {
  assert.equal(clampSum([0, 1, 2], 1, 1), 3);
});

test("empty input sums to zero", () => {
  assert.equal(clampSum([], 0, 1), 0);
});

test("rejects lo > hi", () => {
  assert.throws(() => clampSum([1], 5, 1));
});
