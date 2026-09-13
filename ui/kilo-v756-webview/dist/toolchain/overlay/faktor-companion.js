// Faktor companion panel for the vendored Kilo v7.5.6 webview.
//
// This file is NOT upstream source. It is an additive overlay merged into
// `dist/overlay/` by `apps/vscode/scripts/prepare-vendored-webview.mjs` at
// `prepackage:vsix` time (source-of-truth:
// `ui/kilo-v756-webview/dist/toolchain/overlay/`, a directory the pinned
// toolchain preserves across rebuilds). It consumes the additive `faktor*`
// extension messages emitted by apps/vscode/src/kilo-bridge.ts and posts
// only the strictly validated `faktor*` panel actions back. No frozen Kilo
// message is read, written, or interpreted here.
//
// Plain browser JS, no bundler, no imports. Every value is written with
// textContent (never innerHTML) and every list is bounded before it lands in
// the DOM.

(function () {
  'use strict';

  var TASK = 'faktorTaskState';
  var AGENTS = 'faktorAgents';
  var COCKPIT = 'faktorCockpit';
  var TOURNAMENT = 'faktorTournament';
  var EVIDENCE = 'faktorEvidence';
  var BOARD = 'faktorBoardState';
  var TYPES = [TASK, AGENTS, COCKPIT, TOURNAMENT, EVIDENCE, BOARD];

  var MAX_LINES = 64;
  var MAX_TEXT = 400;
  var MAX_POSTS = 100;
  var MAX_BODY_CHARS = 4096;
  var MAX_SUBJECT_CHARS = 256;
  var MAX_REF_CHARS = 512;
  // Mirrors the runtime's steering-note bound (apps/vscode BRIDGE_LIMITS
  // and faktor_session::MAX_CHILD_CONTROL_NOTE_CHARS): the inline textbox
  // can never emit a note the host/daemon would refuse.
  var MAX_STEER_CHARS = 500;

  var state = {
    task: null,
    agents: null,
    cockpit: null,
    tournament: null,
    evidence: null,
    board: null
  };

  // ---------------------------------------------------------------- helpers

  function text(value, max) {
    if (typeof value !== 'string') {
      return '';
    }
    var out = value.length > (max || MAX_TEXT) ? value.slice(0, max || MAX_TEXT) + '…' : value;
    return out;
  }

  function array(value) {
    return Array.isArray(value) ? value : [];
  }

  function el(tag, className, value) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (value !== undefined) {
      node.textContent = String(value);
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function badge(parent, label, tone) {
    var node = el('span', 'faktor-badge' + (tone ? ' faktor-badge-' + tone : ''), label);
    parent.appendChild(node);
    return node;
  }

  function button(parent, label, className, onClick) {
    var node = el('button', className || 'faktor-btn', label);
    node.type = 'button';
    node.addEventListener('click', function (event) {
      if (event && typeof event.preventDefault === 'function') {
        event.preventDefault();
      }
      onClick();
    });
    parent.appendChild(node);
    return node;
  }

  function section(parent, title, key) {
    var root = el('section', 'faktor-section');
    if (key) {
      root.setAttribute('data-section', key);
    }
    root.appendChild(el('h3', 'faktor-section-title', title));
    parent.appendChild(root);
    return root;
  }

  function line(parent, value, className) {
    parent.appendChild(el('div', className || 'faktor-line', value));
  }

  function api() {
    try {
      if (typeof window.__faktorVsCodeApi === 'function') {
        return window.__faktorVsCodeApi();
      }
    } catch (error) {
      return null;
    }
    return null;
  }

  function post(message) {
    var handle = api();
    if (handle && typeof handle.postMessage === 'function') {
      handle.postMessage(message);
      return true;
    }
    return false;
  }

  function jsonLine(value) {
    if (value === null || value === undefined) {
      return null;
    }
    try {
      var encoded = JSON.stringify(value);
      if (encoded === undefined || encoded === 'null' || encoded === '{}' || encoded === '[]') {
        return null;
      }
      return text(encoded, 240);
    } catch (error) {
      return null;
    }
  }

  // ------------------------------------------------------------ task block

  function renderTask(root) {
    var data = state.task;
    if (!data || data.present === false) {
      var empty = section(root, 'Task');
      line(empty, 'no active task', 'faktor-muted');
      return;
    }
    var node = section(root, 'Task', 'task');
    var head = el('div', 'faktor-row');
    head.appendChild(el('span', 'faktor-strong', text(data.goal, 200) || 'task'));
    if (data.state) {
      badge(head, text(data.state, 48), 'state');
    }
    if (data.phase) {
      badge(head, text(data.phase, 48));
    }
    node.appendChild(head);
    var verification = data.verification;
    if (verification && typeof verification === 'object') {
      line(
        node,
        'verification ' +
          text(verification.status, 32) +
          ' · criteria ' +
          Number(verification.criteriaPassed || 0) +
          '/' +
          Number(verification.criteriaTotal || 0) +
          ' · failed checks ' +
          Number(verification.failedChecks || 0) +
          ' · owed ' +
          Number(verification.owed || 0),
        'faktor-line faktor-muted'
      );
    }
    var criteria = array(data.acceptanceCriteria).slice(0, MAX_LINES);
    if (criteria.length > 0) {
      line(node, 'acceptance', 'faktor-label');
      for (var i = 0; i < criteria.length; i += 1) {
        line(node, '• ' + text(criteria[i]), 'faktor-line');
      }
    }
    var milestones = data.milestones || {};
    var completed = array(milestones.completed).slice(0, MAX_LINES);
    var open = array(milestones.open).slice(0, MAX_LINES);
    if (completed.length + open.length > 0) {
      line(node, 'milestones', 'faktor-label');
      for (var c = 0; c < completed.length; c += 1) {
        line(node, '✓ ' + text(completed[c]), 'faktor-line faktor-done');
      }
      for (var o = 0; o < open.length; o += 1) {
        line(node, '○ ' + text(open[o]), 'faktor-line');
      }
    }
    var tests = data.tests || {};
    var failed = array(tests.failed).slice(0, MAX_LINES);
    if (failed.length > 0) {
      line(node, 'failing tests', 'faktor-label');
      for (var f = 0; f < failed.length; f += 1) {
        line(node, '! ' + text(failed[f]), 'faktor-line faktor-failed');
      }
    }
    var blockers = array(data.blockers).slice(0, MAX_LINES);
    if (blockers.length > 0) {
      line(node, 'blockers', 'faktor-label');
      for (var b = 0; b < blockers.length; b += 1) {
        line(node, '! ' + text(blockers[b]), 'faktor-line faktor-blocked');
      }
    }
    var budget = data.budget;
    if (budget && typeof budget === 'object') {
      line(
        node,
        'spend tokens ' +
          text(String(budget.spentTokens === null || budget.spentTokens === undefined ? '—' : budget.spentTokens), 32) +
          (budget.maxTokens === null || budget.maxTokens === undefined ? '' : ' / ' + text(String(budget.maxTokens), 32)) +
          ' · cost ' +
          text(String(budget.spentCostMicro === undefined ? 0 : budget.spentCostMicro), 32) +
          'µ' +
          (budget.maxCostMicro === null || budget.maxCostMicro === undefined ? '' : ' / ' + text(String(budget.maxCostMicro), 32) + 'µ'),
        'faktor-line faktor-muted'
      );
    }
  }

  // ----------------------------------------------------------- agent cards

  function pixelNode(pixel, agentId) {
    var animation = pixel && pixel.animation ? String(pixel.animation) : 'pixel-waiting';
    var wrap = el('span', 'faktor-pixel ' + animation);
    wrap.setAttribute('data-child', agentId);
    var avatar = (pixel && pixel.avatar) || {};
    var bits = array(avatar.pixels);
    var color = typeof avatar.color === 'string' ? avatar.color : '';
    var accent = typeof avatar.accent === 'string' ? avatar.accent : '';
    for (var i = 0; i < 25; i += 1) {
      var bit = el('span', bits[i] === 1 ? 'faktor-pixel-bit on' : 'faktor-pixel-bit');
      bit.style.color = accent;
      bit.style.backgroundColor = bits[i] === 1 ? color : 'transparent';
      wrap.appendChild(bit);
    }
    return wrap;
  }

  function agentActions(parent, agent) {
    var actions = el('div', 'faktor-actions');
    var st = String(agent.state || '').toLowerCase();
    var terminal = st.indexOf('done') === 0 || st === 'completed' || st.indexOf('cancel') === 0;
    var failed = st.indexOf('fail') === 0;
    var running =
      st === 'running' ||
      st === 'preparing' ||
      st === 'streaming' ||
      st === 'buildingcontext' ||
      st === 'waitingformodel' ||
      st === 'executingtool' ||
      st === 'validating' ||
      st === 'toolrequested';
    var paused = st === 'paused' || st === 'suspended';
    if (running) {
      button(actions, 'Pause', 'faktor-btn', function () {
        post({ type: 'faktorAgentAction', agentId: agent.agentId, action: 'pause' });
      });
    }
    if (paused) {
      button(actions, 'Resume', 'faktor-btn', function () {
        post({ type: 'faktorAgentAction', agentId: agent.agentId, action: 'resume' });
      });
    }
    if (failed) {
      button(actions, 'Retry', 'faktor-btn faktor-btn-primary', function () {
        post({ type: 'faktorAgentAction', agentId: agent.agentId, action: 'retry' });
      });
    }
    if (!terminal) {
      button(actions, 'Cancel', 'faktor-btn', function () {
        post({ type: 'faktorAgentAction', agentId: agent.agentId, action: 'cancel' });
      });
      button(actions, 'Model', 'faktor-btn', function () {
        post({ type: 'faktorAgentAction', agentId: agent.agentId, action: 'model' });
      });
      button(actions, 'Budget', 'faktor-btn', function () {
        post({ type: 'faktorAgentAction', agentId: agent.agentId, action: 'budget' });
      });
      // Steer (P2 UI parity): a BOUNDED inline note posts the exact
      // `faktorAgentAction` body; an empty box posts the bare action and
      // the host prompts instead. The runtime keeps the non-empty/<=500
      // guard, so a hostile/oversized note is refused typed on both the
      // bridge and the host path.
      var steerRow = el('div', 'faktor-steer');
      var steerInput = el('input', 'faktor-input');
      steerInput.type = 'text';
      steerInput.placeholder = 'steer note (optional)';
      steerInput.setAttribute('aria-label', 'steer note');
      steerInput.maxLength = MAX_STEER_CHARS;
      steerRow.appendChild(steerInput);
      button(steerRow, 'Steer', 'faktor-btn', function () {
        var note = typeof steerInput.value === 'string' ? steerInput.value.trim() : '';
        if (note.length === 0) {
          post({ type: 'faktorAgentAction', agentId: agent.agentId, action: 'steer' });
          return;
        }
        post({
          type: 'faktorAgentAction',
          agentId: agent.agentId,
          action: 'steer',
          text: note.slice(0, MAX_STEER_CHARS)
        });
      });
      actions.appendChild(steerRow);
    }
    var next = agent.presentation === 'background' ? 'foreground' : 'background';
    button(
      actions,
      agent.presentation === 'background' ? 'Foreground' : 'Background',
      'faktor-btn',
      function () {
        post({
          type: 'faktorAgentAction',
          agentId: agent.agentId,
          action: 'presentation',
          state: next
        });
      }
    );
    parent.appendChild(actions);
  }

  function renderAgents(root) {
    var data = state.agents;
    var agents = array(data && data.agents);
    var node = section(root, 'Agents', 'agents');
    if (agents.length === 0) {
      line(node, 'none', 'faktor-muted');
      return;
    }
    for (var i = 0; i < agents.length && i < MAX_LINES; i += 1) {
      var agent = agents[i] || {};
      var card = el('article', 'faktor-agent');
      var presentation = agent.presentation === 'background' ? 'background' : 'foreground';
      card.setAttribute('data-presentation', presentation);
      card.setAttribute('data-state', String(agent.state || 'unknown'));
      var head = el('div', 'faktor-row');
      head.appendChild(pixelNode(agent.pixel, agent.agentId));
      head.appendChild(el('span', 'faktor-strong', text(agent.agentId, 80)));
      badge(head, text(agent.state || 'unknown', 40), 'state');
      if (presentation === 'background') {
        badge(head, 'background', 'dim');
      }
      card.appendChild(head);
      var item = agent.itemId ? 'item ' + text(agent.itemId, 80) : null;
      var meta = [];
      if (item) {
        meta.push(item);
      }
      if (agent.worktreeId !== null && agent.worktreeId !== undefined) {
        meta.push('worktree ' + text(String(agent.worktreeId), 24));
      }
      if (agent.ownership) {
        meta.push(text(String(agent.ownership), 64));
      }
      if (agent.model) {
        meta.push('model ' + text(String(agent.model), 80));
      }
      if (agent.provider) {
        meta.push('provider ' + text(String(agent.provider), 80));
      }
      if (agent.reasoning === true || agent.thinking === true) {
        meta.push(agent.thinking === true ? 'thinking' : 'reasoning');
      }
      if (agent.budget !== null && agent.budget !== undefined) {
        meta.push('budget ' + text(String(agent.budget), 32));
      }
      if (meta.length > 0) {
        line(card, meta.join(' · '), 'faktor-line faktor-muted');
      }
      if (agent.goal) {
        line(card, text(agent.goal, 200), 'faktor-line');
      }
      var progress = jsonLine(agent.progress);
      if (progress) {
        line(card, 'progress ' + progress, 'faktor-line faktor-muted');
      }
      var result = jsonLine(agent.result);
      if (result) {
        line(card, 'result ' + result, 'faktor-line faktor-muted');
      }
      var blockers = array(agent.blockers).slice(0, 8);
      for (var b = 0; b < blockers.length; b += 1) {
        line(card, '! ' + text(blockers[b]), 'faktor-line faktor-blocked');
      }
      agentActions(card, agent);
      node.appendChild(card);
    }
  }

  // --------------------------------------------------------- cockpit block

  function renderCockpit(root) {
    var data = state.cockpit;
    var sections = array(data && data.sections);
    var node = section(root, 'Cockpit', 'cockpit');
    if (sections.length === 0) {
      line(node, 'none', 'faktor-muted');
      return;
    }
    for (var i = 0; i < sections.length && i < 32; i += 1) {
      var item = sections[i] || {};
      if (item.key === 'tournament' || item.key === 'evidence' || item.key === 'children') {
        continue;
      }
      var block = el('div', 'faktor-cockpit-section');
      block.setAttribute('data-cockpit', String(item.key || ''));
      block.appendChild(el('div', 'faktor-label', text(item.title || item.key, 80)));
      var lines = array(item.lines).slice(0, MAX_LINES);
      for (var l = 0; l < lines.length; l += 1) {
        line(block, text(lines[l]), 'faktor-line');
      }
      node.appendChild(block);
    }
  }

  // ------------------------------------------------------ tournament block

  function renderTournament(root) {
    var data = state.tournament;
    var node = section(root, 'Tournament', 'tournament');
    if (!data || data.present === false || !data.tournament) {
      line(node, 'none', 'faktor-muted');
      return;
    }
    var tournament = data.tournament;
    var head = el('div', 'faktor-row');
    head.appendChild(el('span', 'faktor-strong', text(tournament.id, 80)));
    badge(head, text(tournament.state, 32), 'state');
    if (tournament.winner) {
      badge(head, 'winner ' + text(tournament.winner, 80), 'winner');
    }
    node.appendChild(head);
    var criteria = array(tournament.criteria).slice(0, MAX_LINES);
    if (criteria.length > 0) {
      line(node, 'criteria', 'faktor-label');
      for (var c = 0; c < criteria.length; c += 1) {
        line(node, '• ' + text(criteria[c]), 'faktor-line');
      }
    }
    var candidates = array(tournament.candidates).slice(0, 16);
    for (var i = 0; i < candidates.length; i += 1) {
      var candidate = candidates[i] || {};
      var card = el('article', 'faktor-candidate');
      var row = el('div', 'faktor-row');
      row.appendChild(el('span', 'faktor-strong', text(candidate.childId, 80)));
      badge(row, text(candidate.state, 32), 'state');
      if (candidate.winner === true) {
        badge(row, 'winner', 'winner');
      }
      card.appendChild(row);
      var verdict =
        candidate.verificationPass === true
          ? 'verification pass'
          : candidate.verificationPass === false
            ? 'verification fail'
            : 'unverified';
      var review = candidate.reviewRank
        ? ' · review ' + text(candidate.reviewRank, 32) + (candidate.reviewer ? ' (' + text(candidate.reviewer, 64) + ')' : '')
        : '';
      line(
        card,
        verdict +
          review +
          ' · cost ' +
          text(String(candidate.costMicro === undefined ? 0 : candidate.costMicro), 32) +
          'µ · wall ' +
          text(String(candidate.wallMs === undefined ? 0 : candidate.wallMs), 32) +
          'ms',
        'faktor-line faktor-muted'
      );
      node.appendChild(card);
    }
    var actions = el('div', 'faktor-actions');
    var decide = button(actions, 'Decide winner', 'faktor-btn faktor-btn-primary', function () {
      post({
        type: 'faktorTournamentAction',
        tournamentId: tournament.id,
        action: 'decide'
      });
    });
    decide.disabled = tournament.canDecide !== true;
    var reasonInput = el('input', 'faktor-input');
    reasonInput.type = 'text';
    reasonInput.placeholder = 'abort reason (optional)';
    reasonInput.setAttribute('aria-label', 'abort reason');
    reasonInput.maxLength = 512;
    actions.appendChild(reasonInput);
    var abort = button(actions, 'Abort', 'faktor-btn', function () {
      post({
        type: 'faktorTournamentAction',
        tournamentId: tournament.id,
        action: 'abort',
        reason: typeof reasonInput.value === 'string' ? reasonInput.value.slice(0, 512) : ''
      });
    });
    abort.disabled = tournament.open !== true;
    node.appendChild(actions);
  }

  // -------------------------------------------------------- evidence block

  function renderEvidence(root) {
    var data = state.evidence;
    var node = section(root, 'Evidence', 'evidence');
    if (!data) {
      line(node, 'none', 'faktor-muted');
      return;
    }
    if (data.mode === 'expanded' && data.evidence) {
      var evidence = data.evidence;
      var head = el('div', 'faktor-row');
      head.appendChild(el('span', 'faktor-strong', 'evidence ' + text(String(evidence.id), 32)));
      if (evidence.truncated === true) {
        badge(head, 'truncated', 'dim');
      }
      node.appendChild(head);
      var pre = el('pre', 'faktor-evidence');
      pre.textContent = text(evidence.text, 20000);
      node.appendChild(pre);
      button(node, 'Back to refs', 'faktor-btn', function () {
        state.evidence = { mode: 'refs', refs: state.evidenceRefs || [] };
        render();
      });
      return;
    }
    var refs = array(data.refs).slice(0, 64);
    if (refs.length === 0) {
      line(node, 'none', 'faktor-muted');
      return;
    }
    for (var i = 0; i < refs.length; i += 1) {
      (function (ref) {
        var row = el('div', 'faktor-row');
        var label = ref && ref.id !== null && ref.id !== undefined ? 'evidence:' + ref.id : text(ref && ref.label, 120);
        var open = button(row, label, 'faktor-btn faktor-btn-link', function () {
          var id = ref && typeof ref.id === 'number' ? ref.id : null;
          if (id !== null) {
            post({ type: 'faktorEvidenceExpand', evidenceId: id });
          }
        });
        open.disabled = !(ref && typeof ref.id === 'number');
        node.appendChild(row);
      })(refs[i]);
    }
  }

  // ----------------------------------------------------------- board block

  function renderBoard(root) {
    var data = state.board;
    var node = section(root, 'Board', 'board');
    if (!data || data.available !== true) {
      line(node, data && data.reason ? text(data.reason, 240) : 'board not exposed by this daemon', 'faktor-muted');
      button(node, 'Refresh', 'faktor-btn', function () {
        post({ type: 'faktorBoardAction', action: 'read' });
      });
      return;
    }
    var head = el('div', 'faktor-row');
    head.appendChild(el('span', 'faktor-label', 'coordination board'));
    if (data.unread !== null && data.unread !== undefined) {
      badge(head, String(Number(data.unread) || 0) + ' unread', Number(data.unread) > 0 ? 'winner' : 'dim');
    }
    if (data.revision !== null && data.revision !== undefined) {
      badge(head, 'rev ' + text(String(data.revision), 24), 'dim');
    }
    node.appendChild(head);
    var posts = array(data.posts).slice(0, MAX_POSTS);
    if (posts.length === 0) {
      line(node, 'no posts', 'faktor-muted');
    }
    for (var i = 0; i < posts.length; i += 1) {
      var entry = posts[i] || {};
      var card = el('article', 'faktor-post');
      var row = el('div', 'faktor-row');
      row.appendChild(el('span', 'faktor-strong', text(entry.author || 'agent', 64)));
      row.appendChild(el('span', 'faktor-muted', text(entry.subject, 120)));
      card.appendChild(row);
      if (entry.body) {
        line(card, text(entry.body, MAX_BODY_CHARS), 'faktor-line');
      }
      var refs = array(entry.refs).slice(0, 32);
      for (var r = 0; r < refs.length; r += 1) {
        line(card, 'ref ' + text(refs[r], MAX_REF_CHARS), 'faktor-line faktor-muted');
      }
      node.appendChild(card);
    }
    var composer = el('div', 'faktor-composer');
    var subject = el('input', 'faktor-input');
    subject.type = 'text';
    subject.placeholder = 'subject';
    subject.setAttribute('aria-label', 'board subject');
    subject.maxLength = MAX_SUBJECT_CHARS;
    composer.appendChild(subject);
    var body = el('textarea', 'faktor-input');
    body.rows = 2;
    body.placeholder = 'post to the run-family board';
    body.setAttribute('aria-label', 'board body');
    composer.appendChild(body);
    button(composer, 'Post', 'faktor-btn', function () {
      var subjectText = typeof subject.value === 'string' ? subject.value.trim() : '';
      var bodyText = typeof body.value === 'string' ? body.value : '';
      if (subjectText.length === 0) {
        return;
      }
      post({
        type: 'faktorBoardAction',
        action: 'post',
        subject: subjectText.slice(0, MAX_SUBJECT_CHARS),
        body: bodyText.slice(0, MAX_BODY_CHARS)
      });
    });
    button(composer, 'Refresh', 'faktor-btn', function () {
      post({ type: 'faktorBoardAction', action: 'read' });
    });
    node.appendChild(composer);
  }

  // ------------------------------------------------------------------ render

  var panel = null;

  function render() {
    if (!panel) {
      return;
    }
    clear(panel);
    panel.appendChild(el('header', 'faktor-panel-title', 'Faktor'));
    renderTask(panel);
    renderAgents(panel);
    renderCockpit(panel);
    renderTournament(panel);
    renderEvidence(panel);
    renderBoard(panel);
  }

  function handle(data) {
    if (!data || TYPES.indexOf(data.type) < 0) {
      return;
    }
    if (data.type === TASK) {
      state.task = data;
    } else if (data.type === AGENTS) {
      state.agents = data;
    } else if (data.type === COCKPIT) {
      state.cockpit = data;
    } else if (data.type === TOURNAMENT) {
      state.tournament = data;
    } else if (data.type === EVIDENCE) {
      if (data.mode === 'refs') {
        state.evidenceRefs = array(data.refs).slice(0, 64);
        state.evidence = { mode: 'refs', refs: state.evidenceRefs };
      } else {
        state.evidence = data;
      }
    } else if (data.type === BOARD) {
      state.board = data;
    }
    render();
  }

  function mount() {
    if (panel && panel.parentNode) {
      return panel;
    }
    panel = el('aside', 'faktor-panel');
    panel.id = 'faktor-companion';
    panel.setAttribute('aria-label', 'Faktor companion');
    var root = document.getElementById('root');
    if (root && root.parentNode && typeof root.parentNode.insertBefore === 'function') {
      var shell = el('div', 'faktor-shell');
      root.parentNode.insertBefore(shell, root);
      shell.appendChild(root);
      shell.appendChild(panel);
    } else if (document.body) {
      document.body.appendChild(panel);
    }
    render();
    return panel;
  }

  window.addEventListener('message', function (event) {
    var data = event && event.data;
    if (data && typeof data.type === 'string') {
      handle(data);
    }
  });

  window.__faktorCompanion = { handle: handle, mount: mount, render: render, types: TYPES };

  if (document.readyState === 'loading' && typeof document.addEventListener === 'function') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
