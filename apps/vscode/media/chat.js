// Faktor chat webview script. Hand-written, dependency-free, no remote
// loads, no eval: it renders the snapshot the extension posts and sends
// typed commands back. All daemon-derived strings go through textContent.
(function () {
  'use strict';

  var vscode = acquireVsCodeApi();

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

  function formatMicro(micro) {
    if (typeof micro !== 'number' || !isFinite(micro)) {
      return '—';
    }
    return '$' + (micro / 1000000).toFixed(4);
  }

  function formatBudget(budget) {
    if (!budget) {
      return 'no budget';
    }
    var parts = [];
    if (budget.spentTokens !== null && budget.spentTokens !== undefined) {
      parts.push(
        'tokens ' +
          budget.spentTokens +
          (budget.maxTokens === null || budget.maxTokens === undefined
            ? ''
            : ' / ' + budget.maxTokens),
      );
    }
    parts.push(
      'cost ' +
        formatMicro(budget.spentCostMicro) +
        (budget.maxCostMicro === null || budget.maxCostMicro === undefined
          ? ''
          : ' / ' + formatMicro(budget.maxCostMicro)),
    );
    if (budget.openReservedMicro > 0) {
      parts.push('reserved ' + formatMicro(budget.openReservedMicro));
    }
    return parts.join(' · ');
  }

  function renderTask(task) {
    var card = byId('task-card');
    if (!task) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    setText('task-state', task.state);
    setText('task-goal', task.goal);

    var milestones = byId('task-milestones');
    clear(milestones);
    if (task.completed.length === 0 && task.open.length === 0) {
      line(milestones, 'milestones: none yet', 'muted');
    } else {
      line(milestones, 'done: ' + (task.completed.join('; ') || '—'));
      line(milestones, 'open: ' + (task.open.join('; ') || '—'));
    }

    var tests = byId('task-tests');
    clear(tests);
    line(tests, 'tests run: ' + (task.testsRun.join(', ') || '—'), 'muted');
    line(
      tests,
      'tests failed: ' + (task.testsFailed.join(', ') || '—'),
      task.testsFailed.length > 0 ? 'error' : 'muted',
    );

    var files = byId('task-files');
    clear(files);
    for (var i = 0; i < task.changedFiles.length && i < 30; i++) {
      line(files, task.changedFiles[i], 'muted');
    }
    if (task.changedFiles.length > 30) {
      line(files, '… ' + (task.changedFiles.length - 30) + ' more', 'muted');
    }
  }

  function renderVerification(view) {
    var node = byId('task-verification');
    if (!node) {
      return;
    }
    clear(node);
    var owed = view && Array.isArray(view.owed) ? view.owed : [];
    var failed = view && Array.isArray(view.failedChecks) ? view.failedChecks : [];
    line(node, 'verification owed: ' + owed.length, owed.length > 0 ? 'warn' : 'muted');
    for (var i = 0; i < owed.length && i < 10; i++) {
      line(
        node,
        owed[i].tool + ' ' + owed[i].status + (owed[i].effectStatus ? ' (' + owed[i].effectStatus + ')' : ''),
        'muted',
      );
    }
    line(node, 'failed checks: ' + failed.length, failed.length > 0 ? 'error' : 'muted');
    for (var j = 0; j < failed.length && j < 10; j++) {
      line(node, failed[j].detail || failed[j].id, 'error');
    }
  }

  function renderUsage(usage) {
    var node = byId('task-budget');
    if (!node) {
      return;
    }
    if (!usage) {
      node.textContent = '';
      return;
    }
    var parts = ['tokens ' + usage.tokens];
    parts.push(
      'cost ' + formatMicro(usage.spentMicro) + (usage.maxMicro === null ? '' : ' / ' + formatMicro(usage.maxMicro)),
    );
    if (usage.openMicro > 0) {
      parts.push('reserved ' + formatMicro(usage.openMicro));
    }
    if (usage.truncated) {
      parts.push('(truncated)');
    }
    node.textContent = parts.join(' · ');
  }

  function agentButton(agent, action, label) {
    var button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', function () {
      vscode.postMessage({ type: 'agentControl', agentId: agent.agentId, action: action });
    });
    return button;
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
    for (var i = 0; i < agents.length; i++) {
      var agent = agents[i];
      var item = document.createElement('li');
      var head = document.createElement('div');
      head.className = 'agent-head';
      head.textContent =
        agent.kind + ' ' + agent.agentId + ' · ' + agent.state + (agent.model ? ' · ' + agent.model : '');
      item.appendChild(head);
      if (agent.goal) {
        var goal = document.createElement('div');
        goal.className = 'muted';
        goal.textContent = agent.goal;
        item.appendChild(goal);
      }
      if (agent.kind === 'child') {
        var controls = document.createElement('div');
        controls.className = 'agent-controls';
        controls.appendChild(agentButton(agent, 'pause', 'Pause'));
        controls.appendChild(agentButton(agent, 'resume', 'Resume'));
        controls.appendChild(agentButton(agent, 'steer', 'Steer'));
        controls.appendChild(agentButton(agent, 'model', 'Model'));
        controls.appendChild(agentButton(agent, 'budget', 'Budget'));
        controls.appendChild(agentButton(agent, 'cancel', 'Cancel'));
        item.appendChild(controls);
      }
      list.appendChild(item);
    }
  }

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
    renderTask(snapshot.task);
    renderVerification(snapshot.verification);
    renderUsage(snapshot.usage);
    if (!snapshot.usage && snapshot.task) {
      byId('task-budget').textContent = formatBudget(snapshot.task.budget);
    }
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
    } else if (message.type === 'notice') {
      showNotice(message.level, message.message);
    }
  });

  byId('composer').addEventListener('submit', function (event) {
    event.preventDefault();
    var goal = byId('goal').value.trim();
    if (!goal) {
      return;
    }
    byId('goal').value = '';
    vscode.postMessage({ type: 'sendGoal', goal: goal });
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
