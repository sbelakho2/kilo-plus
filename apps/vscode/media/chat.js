// Faktor chat webview script. Hand-written, dependency-free, no remote
// loads, no eval: it renders the snapshot the extension posts and sends
// typed commands back. All daemon-derived strings go through textContent.
// Composer draft policy comes from media/composer-state.js (loaded first).
(function () {
  'use strict';

  var vscode = acquireVsCodeApi();
  var composerPolicy = (typeof FaktorComposer !== 'undefined' && FaktorComposer) || {
    afterStart: function (draft) {
      return String(draft == null ? '' : draft);
    },
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function setText(id, value) {
    var node = byId(id);
    if (!node) {
      return;
    }
    if (value === null || value === undefined || value === '') {
      node.textContent = '—';
      return;
    }
    node.textContent = String(value);
  }

  function line(parent, value, className) {
    var node = document.createElement('div');
    if (className) {
      node.className = className;
    }
    node.textContent = String(value);
    parent.appendChild(node);
    return node;
  }

  // ------------------------------------------------------------- cockpit

  function renderCockpit(view, sections) {
    var node = byId('cockpit');
    clear(node);
    if (!view || !Array.isArray(sections) || sections.length === 0) {
      return;
    }
    for (var i = 0; i < sections.length; i++) {
      var section = sections[i];
      var block = document.createElement('section');
      block.className = 'cockpit-section' + (section.present ? '' : ' cockpit-empty');
      var head = document.createElement('h3');
      head.textContent = section.title;
      block.appendChild(head);
      var lines = Array.isArray(section.lines) ? section.lines : [];
      for (var j = 0; j < lines.length; j++) {
        if (section.key === 'evidence') {
          renderEvidenceLine(block, section, lines[j], j);
        } else {
          line(block, lines[j], 'muted');
        }
      }
      // State-gated tournament controls: a disabled action is rendered
      // disabled and NEVER posts (the engine's settle rule is the gate).
      var actions = Array.isArray(section.actions) ? section.actions : [];
      if (section.key === 'tournament' && view && view.tournament && actions.length > 0) {
        var controls = document.createElement('div');
        controls.className = 'cockpit-actions';
        for (var k = 0; k < actions.length; k++) {
          (function (action) {
            var button = document.createElement('button');
            button.type = 'button';
            button.textContent = action.label;
            button.disabled = action.enabled !== true;
            button.addEventListener('click', function () {
              if (action.enabled !== true || !view.tournament) {
                return;
              }
              vscode.postMessage({
                type: 'tournamentControl',
                tournamentId: view.tournament.id,
                action: action.key,
              });
            });
            controls.appendChild(button);
          })(actions[k]);
        }
        block.appendChild(controls);
      }
      node.appendChild(block);
    }
  }

  function renderEvidenceLine(block, section, label, index) {
    var refs = Array.isArray(section.evidence) ? section.evidence : [];
    var ref = refs[index];
    var row = document.createElement('div');
    row.className = 'evidence';
    var text = document.createElement('span');
    text.className = 'muted';
    text.textContent = label;
    row.appendChild(text);
    if (ref && typeof ref.id === 'number' && isFinite(ref.id)) {
      var id = ref.id;
      row.setAttribute('data-evidence', String(id));
      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'View evidence #' + id;
      button.addEventListener('click', function () {
        vscode.postMessage({ type: 'retrieveEvidence', evidenceId: id });
      });
      row.appendChild(button);
    }
    block.appendChild(row);
  }

  function renderTask(task, view) {
    var card = byId('task-card');
    if (!task && !view) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    setText('task-state', task ? task.state : '—');
    setText('task-goal', task ? task.goal : '—');
    renderCompletion(task);
  }

  /**
   * The durable completion-contract block. The step rows are exactly what
   * the host reports: `daemon` rows when the serving daemon exposes the
   * read, `derived` rows projected from the durable task-run state, or an
   * explicit `unavailable` (never fabricated as success).
   */
  function renderCompletion(task) {
    var node = byId('task-completion');
    if (!node) {
      return;
    }
    clear(node);
    var completion = task && task.completion ? task.completion : null;
    if (!completion) {
      return;
    }
    var head = document.createElement('div');
    head.className = 'completion-head';
    head.textContent = 'Completion contract · status source: ' + String(completion.source || 'unknown');
    node.appendChild(head);
    var steps = Array.isArray(completion.steps) ? completion.steps : [];
    for (var i = 0; i < steps.length; i++) {
      var step = steps[i];
      var detail = step.detail ? ' — ' + String(step.detail) : '';
      line(node, '[' + String(step.status) + '] ' + String(step.step) + detail, 'muted');
    }
    if (completion.reason) {
      line(node, String(completion.reason), 'warn');
    }
  }

  // -------------------------------------------------------- agent panel

  function agentButton(agent, action, label) {
    var button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', function () {
      vscode.postMessage({ type: 'agentControl', agentId: agent.agentId, action: action });
    });
    return button;
  }

  /** The durable presentation tag (absent = foreground, per the native v1 contract). */
  function presentationOf(agent) {
    return agent && agent.presentation === 'background' ? 'background' : 'foreground';
  }

  /** The foreground/background toggle: posts the EXACT next durable state. */
  function presentationButton(agent) {
    var background = presentationOf(agent) === 'background';
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'presentation-toggle';
    button.textContent = background ? 'Foreground' : 'Background';
    button.addEventListener('click', function () {
      vscode.postMessage({
        type: 'agentControl',
        agentId: agent.agentId,
        action: 'presentation',
        state: background ? 'foreground' : 'background',
      });
    });
    return button;
  }

  /** The deterministic pixel sprite as an inline SVG (no external assets). */
  function pixelSprite(agent) {
    var pixel = agent.pixel;
    if (!pixel || !pixel.avatar || !Array.isArray(pixel.avatar.pixels)) {
      return null;
    }
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 5 5');
    svg.setAttribute('width', '22');
    svg.setAttribute('height', '22');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'pixel ' + String(pixel.animation || 'pixel-waiting'));
    var pixels = pixel.avatar.pixels;
    for (var y = 0; y < 5; y++) {
      for (var x = 0; x < 5; x++) {
        if (pixels[y * 5 + x] !== 1) {
          continue;
        }
        var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', String(x));
        rect.setAttribute('y', String(y));
        rect.setAttribute('width', '1');
        rect.setAttribute('height', '1');
        rect.setAttribute('fill', pixel.avatar.color);
        svg.appendChild(rect);
      }
    }
    var leftEye = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    leftEye.setAttribute('x', '1');
    leftEye.setAttribute('y', '1');
    leftEye.setAttribute('width', '1');
    leftEye.setAttribute('height', '1');
    leftEye.setAttribute('fill', pixel.avatar.accent);
    svg.appendChild(leftEye);
    var rightEye = leftEye.cloneNode();
    rightEye.setAttribute('x', '3');
    svg.appendChild(rightEye);
    return svg;
  }

  function joinBounded(values, limit) {
    if (!Array.isArray(values) || values.length === 0) {
      return '—';
    }
    var shown = values.slice(0, limit || 12).map(function (value) {
      return typeof value === 'string' ? value : JSON.stringify(value);
    });
    if (values.length > shown.length) {
      shown.push('… ' + (values.length - shown.length) + ' more');
    }
    return shown.join('; ');
  }

  function compactJson(value) {
    if (value === null || value === undefined) {
      return '—';
    }
    try {
      var text = JSON.stringify(value);
      return text === undefined ? '—' : text;
    } catch (error) {
      return '—';
    }
  }

  function renderAgents(agents) {
    var card = byId('agents-card');
    var list = byId('agent-list');
    clear(list);
    if (!agents || agents.length === 0) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    // Background children are dimmed AND grouped/tucked after every
    // foreground entry; the order derives only from the durable
    // presentation field, never from a state heuristic.
    var ordered = [];
    var backgroundCount = 0;
    var i;
    for (i = 0; i < agents.length; i++) {
      if (agents[i].kind === 'child' && presentationOf(agents[i]) === 'background') {
        backgroundCount += 1;
      } else {
        ordered.push(agents[i]);
      }
    }
    var grouped = ordered.slice();
    if (backgroundCount > 0) {
      grouped.push({ groupLabel: 'Background (' + backgroundCount + ') — dimmed', background: true });
    }
    for (i = 0; i < agents.length; i++) {
      if (agents[i].kind === 'child' && presentationOf(agents[i]) === 'background') {
        grouped.push(agents[i]);
      }
    }

    for (i = 0; i < grouped.length; i++) {
      var entry = grouped[i];
      if (entry.groupLabel) {
        var groupItem = document.createElement('li');
        groupItem.className = 'agent-group-label';
        groupItem.textContent = entry.groupLabel;
        list.appendChild(groupItem);
        continue;
      }
      var agent = entry;
      var background = agent.kind === 'child' && presentationOf(agent) === 'background';
      var item = document.createElement('li');
      item.className =
        'agent agent-' +
        (agent.kind === 'child' ? 'child' : 'self') +
        (background ? ' agent-background' : '');

      var head = document.createElement('div');
      head.className = 'agent-head';
      var sprite = pixelSprite(agent);
      if (sprite) {
        head.appendChild(sprite);
      }
      var title = document.createElement('span');
      title.className = 'agent-title';
      title.textContent =
        agent.kind +
        ' ' +
        agent.agentId +
        ' · ' +
        agent.state +
        (background ? ' · background' : '') +
        (agent.model ? ' · ' + agent.model : '') +
        (agent.provider ? ' · ' + agent.provider : '');
      head.appendChild(title);
      item.appendChild(head);

      if (agent.goal) {
        line(item, agent.goal, 'muted');
      }
      var identity = [
        agent.itemId ? 'item ' + agent.itemId + (agent.itemKind ? ' (' + agent.itemKind + ')' : '') : null,
        agent.itemIds && agent.itemIds.length > 0 ? 'items ' + agent.itemIds.join(', ') : null,
        agent.worktreeId !== null && agent.worktreeId !== undefined ? 'worktree ' + agent.worktreeId : null,
        agent.sessionId !== null && agent.sessionId !== undefined ? 'session ' + agent.sessionId : null,
        agent.ownership ? 'ownership ' + agent.ownership : null,
        agent.budget !== null && agent.budget !== undefined ? 'budget ' + agent.budget : null,
        agent.reasoning !== null && agent.reasoning !== undefined
          ? 'reasoning ' + (agent.reasoning ? 'yes' : 'no')
          : null,
        agent.thinking !== null && agent.thinking !== undefined
          ? 'thinking ' + (agent.thinking ? 'yes' : 'no')
          : null,
      ].filter(function (part) {
        return part !== null;
      });
      line(item, identity.join(' · ') || 'identity —', 'muted');
      line(item, 'capabilities: ' + joinBounded(agent.capabilities, 10), 'muted');
      line(item, 'progress: ' + compactJson(agent.progress), 'muted');
      line(item, 'result: ' + compactJson(agent.result), 'muted');
      if (agent.blockers && agent.blockers.length > 0) {
        line(item, 'blockers: ' + agent.blockers.join('; '), 'warn');
      }

      if (agent.kind === 'child') {
        var controls = document.createElement('div');
        controls.className = 'agent-controls';
        controls.appendChild(presentationButton(agent));
        controls.appendChild(agentButton(agent, 'pause', 'Pause'));
        controls.appendChild(agentButton(agent, 'resume', 'Resume'));
        controls.appendChild(agentButton(agent, 'steer', 'Steer'));
        controls.appendChild(agentButton(agent, 'model', 'Model'));
        controls.appendChild(agentButton(agent, 'budget', 'Budget'));
        controls.appendChild(agentButton(agent, 'retry', 'Retry'));
        controls.appendChild(agentButton(agent, 'cancel', 'Cancel'));
        item.appendChild(controls);
      }
      list.appendChild(item);
    }
  }

  // ----------------------------------------------------------- transcript

  function evidenceIdOf(artifact) {
    if (typeof artifact !== 'string') {
      return null;
    }
    var match = /^(?:evidence:)?(\d+)$/.exec(artifact.trim());
    return match ? Number(match[1]) : null;
  }

  function renderTool(container, tool) {
    var row = document.createElement('div');
    row.className = 'tool';
    var head = document.createElement('div');
    head.className = 'tool-head';
    var exitText =
      tool.exitCode === null || tool.exitCode === undefined ? '' : ' · exit ' + tool.exitCode;
    head.textContent = tool.name + ' [' + tool.state + ']' + exitText;
    row.appendChild(head);

    if (tool.excerpt) {
      var excerpt = document.createElement('pre');
      excerpt.className = 'excerpt';
      excerpt.textContent = tool.excerpt;
      row.appendChild(excerpt);
    }

    var evidenceId = evidenceIdOf(tool.artifact);
    if (evidenceId !== null) {
      var holder = document.createElement('div');
      holder.className = 'evidence';
      holder.setAttribute('data-evidence', String(evidenceId));
      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'View evidence #' + evidenceId;
      button.addEventListener('click', function () {
        vscode.postMessage({ type: 'retrieveEvidence', evidenceId: evidenceId });
      });
      holder.appendChild(button);
      row.appendChild(holder);
    }
    container.appendChild(row);
  }

  function renderEntry(entry) {
    var wrapper = document.createElement('article');
    wrapper.className = 'entry entry-' + (entry.role === 'user' ? 'user' : 'assistant');

    var head = document.createElement('div');
    head.className = 'entry-head';
    head.textContent = entry.role + ' #' + entry.seq;
    wrapper.appendChild(head);

    if (entry.text) {
      var text = document.createElement('pre');
      text.className = 'entry-text';
      text.textContent = entry.text;
      wrapper.appendChild(text);
    }
    if (entry.reasoning) {
      var details = document.createElement('details');
      var summary = document.createElement('summary');
      summary.textContent = 'Reasoning';
      details.appendChild(summary);
      var reasoning = document.createElement('pre');
      reasoning.className = 'reasoning';
      reasoning.textContent = entry.reasoning;
      details.appendChild(reasoning);
      wrapper.appendChild(details);
    }
    if (entry.summary) {
      var summaryLine = document.createElement('pre');
      summaryLine.className = 'entry-summary';
      summaryLine.textContent = entry.summary;
      wrapper.appendChild(summaryLine);
    }
    for (var i = 0; i < entry.tools.length; i++) {
      renderTool(wrapper, entry.tools[i]);
    }
    return wrapper;
  }

  function renderTranscript(entries) {
    var container = byId('entries');
    clear(container);
    if (!entries || entries.length === 0) {
      line(container, 'No messages yet.', 'muted');
      return;
    }
    for (var i = 0; i < entries.length; i++) {
      container.appendChild(renderEntry(entries[i]));
    }
    container.scrollTop = container.scrollHeight;
  }

  function setDaemon(status, detail) {
    var dot = byId('daemon-dot');
    dot.className = 'dot dot-' + (status || 'stopped');
    var text = status || 'stopped';
    if (detail) {
      text += ' — ' + detail;
    }
    setText('daemon-text', text);
  }

  function renderSnapshot(snapshot) {
    if (!snapshot) {
      return;
    }
    setDaemon(snapshot.daemon, snapshot.daemonDetail);
    setText('session-title', snapshot.session ? snapshot.session.title : 'none');
    setText('machine-label', snapshot.machineLabel || snapshot.machineState);
    setText('stream-status', snapshot.streamStatus);
    renderTask(snapshot.task, snapshot.cockpit);
    renderCockpit(snapshot.cockpit, snapshot.cockpitSections);
    renderAgents(snapshot.agents);
    renderTranscript(snapshot.transcript);
    if (snapshot.lastError) {
      showNotice('error', snapshot.lastError);
    }
  }

  function showNotice(level, message) {
    var notices = byId('notices');
    var node = document.createElement('div');
    node.className = level === 'error' ? 'notice error' : 'notice';
    node.textContent = String(message);
    notices.appendChild(node);
    while (notices.childNodes.length > 4) {
      notices.removeChild(notices.firstChild);
    }
    setTimeout(function () {
      if (node.parentNode === notices) {
        notices.removeChild(node);
      }
    }, 8000);
  }

  function deliverEvidence(id, text, truncated) {
    var holders = document.querySelectorAll('[data-evidence="' + id + '"]');
    for (var i = 0; i < holders.length; i++) {
      var holder = holders[i];
      clear(holder);
      var pre = document.createElement('pre');
      pre.className = 'excerpt';
      pre.textContent = truncated ? text + '\n… (truncated)' : text;
      holder.appendChild(pre);
    }
  }

  window.addEventListener('message', function (event) {
    var message = event.data || {};
    if (message.type === 'snapshot') {
      renderSnapshot(message.snapshot);
    } else if (message.type === 'evidence') {
      deliverEvidence(message.id, message.text, message.truncated);
    } else if (message.type === 'startResult') {
      var goalNode = byId('goal');
      if (goalNode) {
        goalNode.value = composerPolicy.afterStart(
          goalNode.value,
          message.goal,
          message.ok === true,
        );
      }
      // The completion contract is per task start: a successful ack resets
      // the checkboxes; a failure keeps them for the retry.
      if (message.ok === true) {
        clearCompletionControls();
      }
    } else if (message.type === 'notice') {
      showNotice(message.level, message.message);
    }
  });

  function completionContractFromControls() {
    var commit = byId('contract-commit');
    var push = byId('contract-push');
    var pr = byId('contract-pr');
    if (!commit || !push || !pr) {
      return null;
    }
    if (!commit.checked && !push.checked && !pr.checked) {
      return null;
    }
    return {
      include_commit: commit.checked === true,
      include_push: push.checked === true,
      include_pr: pr.checked === true,
    };
  }

  function clearCompletionControls() {
    var ids = ['contract-commit', 'contract-push', 'contract-pr'];
    for (var i = 0; i < ids.length; i++) {
      var node = byId(ids[i]);
      if (node) {
        node.checked = false;
      }
    }
  }

  byId('composer').addEventListener('submit', function (event) {
    event.preventDefault();
    var goal = byId('goal').value.trim();
    if (!goal) {
      return;
    }
    // Draft preservation: the goal is NOT cleared here. The extension posts
    // a startResult; only a successful start clears the (unchanged) draft.
    // The Task-mode completion contract rides only THIS task start; plain
    // chat never carries it.
    var contract = completionContractFromControls();
    var message = { type: 'sendGoal', goal: goal };
    if (contract) {
      message.completionContract = contract;
    }
    vscode.postMessage(message);
  });
  byId('btn-new-task').addEventListener('click', function () {
    vscode.postMessage({ type: 'newTask' });
  });
  byId('btn-cancel-run').addEventListener('click', function () {
    vscode.postMessage({ type: 'cancelRun' });
  });
  byId('btn-refresh').addEventListener('click', function () {
    vscode.postMessage({ type: 'refresh' });
  });
  byId('btn-start').addEventListener('click', function () {
    vscode.postMessage({ type: 'startDaemon' });
  });
  byId('btn-stop').addEventListener('click', function () {
    vscode.postMessage({ type: 'stopDaemon' });
  });

  vscode.postMessage({ type: 'ready' });
})();
