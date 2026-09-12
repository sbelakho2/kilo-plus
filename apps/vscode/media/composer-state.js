// Composer draft policy, shared by the built-in webview and the selftest.
//
// Defect: the composer cleared the goal textarea at submit time, so a
// refused start (4xx/5xx/validation) lost the user's draft. The draft is
// now cleared ONLY after the extension acknowledges a successful start,
// and only when the textarea still holds the submitted goal (a newer
// draft typed while the request was in flight is never clobbered).
//
// Loaded as a classic script in the webview (defines `FaktorComposer`) and
// importable from Node for tests via `module.exports`.
(function (root) {
  'use strict';

  function normalize(value) {
    return String(value === null || value === undefined ? '' : value).trim();
  }

  /**
   * The textarea value after a start result.
   * - success and the box still holds the submitted goal: cleared;
   * - success but the user typed something new: the new draft is kept;
   * - any failure: the exact draft is retained.
   */
  function afterStart(currentDraft, submittedGoal, ok) {
    if (!ok) {
      return String(currentDraft === null || currentDraft === undefined ? '' : currentDraft);
    }
    return normalize(currentDraft) === normalize(submittedGoal)
      ? ''
      : String(currentDraft === null || currentDraft === undefined ? '' : currentDraft);
  }

  /** True while the composer should keep showing the draft. */
  function keepDraft(currentDraft, submittedGoal, ok) {
    return afterStart(currentDraft, submittedGoal, ok).length > 0;
  }

  var api = { afterStart: afterStart, keepDraft: keepDraft };
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.FaktorComposer = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : null);
