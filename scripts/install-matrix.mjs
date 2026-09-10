#!/usr/bin/env node
// Installation matrix (audit 90): installs/verifies every built artifact from
// target/certification/artifacts.json into a clean temporary prefix on THIS
// host and emits target/certification/install-matrix.json with per-artifact
// pass/fail plus evidence. Exits non-zero on any verification failure.
//
// What is verified for real here:
//   daemon-bundle    sha256 + size against the manifest, tar layout, extract
//                    into a clean prefix, bundle-internal checksums.txt,
//                    `faktor-cli --version`, and
//                    `faktor-cli doctor --data-dir <fresh tmp dir>`.
//   vsix             sha256 + size, zip structure (extension/package.json,
//                    extension.vsixmanifest), package.json parses and its
//                    `main` entry exists in the archive, `contributes`
//                    non-empty.
//   jetbrains-plugin sha256 + size, zip structure, plugin.xml inside the
//                    bundled frontend jar with <id> and <version>.
//
// Residual (CI-only, recorded in the report, never claimed as done here):
// launching a real VS Code / JetBrains IDE and installing the artifact there
// requires an IDE host, so those steps are owned by the CI pr-lane.
//
// Env:
//   MATRIX_ARTIFACTS   manifest to verify (default target/certification/artifacts.json)
//   MATRIX_OUT_DIR     report dir (default target/certification)
//   MATRIX_REQUIRE     colon-separated kinds that must verify (default daemon-bundle)
//   TAMPER=1           self-test: copy an artifact, flip one byte, require this
//                      same verifier to reject it (exits 0 only on rejection)

import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { basename, dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
process.chdir(ROOT);

const OUT_DIR = process.env.MATRIX_OUT_DIR || 'target/certification';
const ARTIFACTS = process.env.MATRIX_ARTIFACTS || join(OUT_DIR, 'artifacts.json');
const REQUIRE = (process.env.MATRIX_REQUIRE || 'daemon-bundle').split(':').filter(Boolean);
const TAMPER = process.env.TAMPER === '1';

const RESIDUAL = [
  'vscode-ide-install: `code --install-extension <vsix>` needs an IDE/CLI host; owned by the CI pr-lane, not claimed by this host run',
  'jetbrains-ide-install: sandbox/IDE install of the plugin zip needs an IDE host; owned by the CI pr-lane, not claimed by this host run',
];

function abs(p) {
  return isAbsolute(p) ? p : resolve(ROOT, p);
}

function sha256(file) {
  return createHash('sha256').update(readFileSync(file)).digest('hex');
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
    ...opts,
  });
  return {
    status: r.status === null ? 127 : r.status,
    stdout: r.stdout || '',
    stderr: r.stderr || (r.error ? String(r.error.message) : ''),
  };
}

function check(checks, name, ok, detail) {
  checks.push({ name, status: ok ? 'pass' : 'fail', detail: String(detail) });
  return ok;
}

function firstLine(text) {
  return (text || '').split('\n').map((l) => l.trim()).find(Boolean) || '';
}

function writeJson(file, value) {
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

// ---------------------------------------------------------------------------
// Per-kind verification.
// ---------------------------------------------------------------------------
function verifyDaemon(file, result) {
  const checks = result.checks;
  const tmp = mkdtempSync(join(tmpdir(), 'faktor-matrix-daemon-'));
  try {
    const list = run('tar', ['-tzf', file]);
    if (!check(checks, 'tar-list', list.status === 0, `tar -tzf exit=${list.status} ${firstLine(list.stderr)}`)) {
      return;
    }
    const entries = list.stdout.split('\n').map((s) => s.trim()).filter(Boolean);
    result.evidence.entries = entries.length;
    const binEntry = entries.find((e) => e === 'bin/faktor-cli' || e.endsWith('/bin/faktor-cli'));
    check(checks, 'bundle-layout', Boolean(binEntry), binEntry ? `binary entry: ${binEntry}` : 'no bin/faktor-cli entry in tar');
    const sumsEntry = entries.find((e) => e === 'checksums.txt' || e.endsWith('/checksums.txt'));
    check(checks, 'bundle-checksums-present', Boolean(sumsEntry), sumsEntry || 'no checksums.txt entry in tar');
    if (!binEntry) return;

    const prefix = join(tmp, 'prefix');
    mkdirSync(prefix, { recursive: true });
    const extract = run('tar', ['-xzf', file, '-C', prefix]);
    if (!check(checks, 'extract', extract.status === 0, `clean prefix ${prefix}; exit=${extract.status} ${firstLine(extract.stderr)}`)) {
      return;
    }
    const bin = join(prefix, binEntry);
    if (!existsSync(bin)) {
      check(checks, 'executable', false, `${bin} missing after extract`);
      return;
    }
    const mode = statSync(bin).mode & 0o111;
    if (!check(checks, 'executable', mode !== 0, `${bin} mode=0o${(statSync(bin).mode & 0o777).toString(8)}`)) return;

    if (sumsEntry) {
      const fields = readFileSync(join(prefix, sumsEntry), 'utf8').trim().split(/\s+/);
      const recorded = fields[0];
      const actual = sha256(bin);
      check(checks, 'bundle-internal-sha256', recorded === actual, `recorded=${recorded} actual=${actual}`);
    }

    const version = run(bin, ['--version']);
    check(checks, 'binary-version', version.status === 0, `${(version.stdout || version.stderr).trim()} (exit=${version.status})`);
    result.evidence.version = (version.stdout || '').trim();

    const dataDir = join(tmp, 'data');
    mkdirSync(dataDir, { recursive: true });
    const doctor = run(bin, ['doctor', '--data-dir', dataDir]);
    const out = `${doctor.stdout}${doctor.stderr}`;
    const passedLine = out.split('\n').map((l) => l.trim()).find((l) => l.includes('doctor:'));
    const ok = doctor.status === 0 && out.includes('doctor: all checks passed');
    check(checks, 'doctor-clean-prefix', ok, `exit=${doctor.status}; ${passedLine || firstLine(out)}; data-dir=${dataDir}`);
    result.evidence.doctor = { exit: doctor.status, line: passedLine || '', data_dir: dataDir };
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

function verifyVsix(file, result) {
  const checks = result.checks;
  const list = run('unzip', ['-Z1', file]);
  if (!check(checks, 'unzip-list', list.status === 0, `unzip -Z1 exit=${list.status} ${firstLine(list.stderr)}`)) {
    return;
  }
  const entries = list.stdout.split('\n').map((s) => s.trim()).filter(Boolean);
  result.evidence.entries = entries.length;
  check(checks, 'package-json', entries.includes('extension/package.json'), 'extension/package.json');
  check(checks, 'vsixmanifest', entries.includes('extension.vsixmanifest'), 'extension.vsixmanifest');

  const raw = run('unzip', ['-p', file, 'extension/package.json']);
  if (!check(checks, 'package-json-extract', raw.status === 0 && raw.stdout.trim().length > 0, `unzip -p exit=${raw.status}`)) {
    return;
  }
  let pkg;
  try {
    pkg = JSON.parse(raw.stdout);
    check(checks, 'package-json-parses', true, 'valid JSON');
  } catch (e) {
    check(checks, 'package-json-parses', false, String(e.message));
    return;
  }
  result.evidence.package = {
    name: pkg.name,
    version: pkg.version,
    publisher: pkg.publisher,
    engines: pkg.engines,
    main: pkg.main,
  };
  const main = String(pkg.main || '').replace(/^\.?\//, '');
  if (!check(checks, 'main-entry', Boolean(main) && entries.includes(`extension/${main}`), `main=${pkg.main} -> extension/${main}`)) {
    return;
  }
  const contributes = pkg.contributes && typeof pkg.contributes === 'object' ? pkg.contributes : null;
  const commands = contributes && Array.isArray(contributes.commands) ? contributes.commands : [];
  const views = contributes && contributes.views ? Object.keys(contributes.views) : [];
  check(checks, 'contributes', Boolean(contributes) && Object.keys(contributes).length > 0,
    contributes ? `keys=${Object.keys(contributes).join(',')}` : 'no contributes block');
  check(checks, 'contributes-commands', commands.length > 0, `commands=${commands.length}`);
  check(checks, 'contributes-views', views.length > 0, `views containers=${views.join(',')}`);
  result.evidence.contributes = { commands: commands.length, views };
}

function verifyJetbrains(file, result) {
  const checks = result.checks;
  const list = run('unzip', ['-Z1', file]);
  if (!check(checks, 'unzip-list', list.status === 0, `unzip -Z1 exit=${list.status} ${firstLine(list.stderr)}`)) {
    return;
  }
  const entries = list.stdout.split('\n').map((s) => s.trim()).filter(Boolean);
  result.evidence.entries = entries.length;
  const jars = entries.filter((e) => /\.jar$/i.test(e));
  if (!check(checks, 'jars-present', jars.length > 0, `jars=${jars.length}: ${jars.join(', ')}`)) {
    return;
  }

  const tmp = mkdtempSync(join(tmpdir(), 'faktor-matrix-jb-'));
  try {
    let xml = '';
    let source = '';
    const direct = entries.find((e) => /(^|\/)META-INF\/plugin\.xml$/.test(e));
    if (direct) {
      const r = run('unzip', ['-p', file, direct]);
      if (r.status === 0) {
        xml = r.stdout;
        source = `${basename(file)}!${direct}`;
      }
    }
    if (!xml) {
      for (const jar of jars) {
        const extract = run('unzip', ['-o', '-q', file, jar, '-d', tmp]);
        if (extract.status !== 0) continue;
        const jarPath = join(tmp, jar);
        const inner = run('unzip', ['-p', jarPath, 'META-INF/plugin.xml']);
        if (inner.status === 0 && inner.stdout.includes('<idea-plugin')) {
          xml = inner.stdout;
          source = `${basename(file)}!${jar}!META-INF/plugin.xml`;
          break;
        }
      }
    }
    if (!check(checks, 'plugin-xml', Boolean(xml), source || 'plugin.xml not found in zip or bundled jars')) {
      return;
    }
    result.evidence.plugin_xml_source = source;
    const id = (xml.match(/<id>([^<]+)<\/id>/) || [])[1] || '';
    const version = (xml.match(/<version>([^<]+)<\/version>/) || [])[1] || '';
    check(checks, 'plugin-id', id.length > 0, `id=${id || '(missing)'}`);
    check(checks, 'plugin-version', version.length > 0, `version=${version || '(missing)'}`);
    result.evidence.plugin = { id, version };
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

const VERIFIERS = {
  'daemon-bundle': verifyDaemon,
  vsix: verifyVsix,
  'jetbrains-plugin': verifyJetbrains,
};

function verifyArtifact(a) {
  const result = {
    name: a.name,
    kind: a.kind,
    status: 'fail',
    path: a.path,
    sha256: a.sha256,
    size: a.size,
    checks: [],
    evidence: {},
  };
  const file = abs(String(a.path || ''));
  if (!check(result.checks, 'present', Boolean(a.path) && existsSync(file), file)) return result;
  const st = statSync(file);
  check(result.checks, 'size', st.size === a.size, `recorded=${a.size} actual=${st.size}`);
  const actual = sha256(file);
  check(result.checks, 'sha256', actual === a.sha256, `recorded=${a.sha256} actual=${actual}`);
  result.evidence.sha256 = actual;
  if (result.checks.some((c) => c.status === 'fail')) return result;
  const verifier = VERIFIERS[a.kind];
  if (!verifier) {
    check(result.checks, 'kind', false, `unknown kind '${a.kind}': no install verification implemented`);
    return result;
  }
  verifier(file, result);
  result.status = result.checks.every((c) => c.status === 'pass') ? 'pass' : 'fail';
  return result;
}

// ---------------------------------------------------------------------------
// TAMPER self-test: flip one byte in a copy and require rejection.
// ---------------------------------------------------------------------------
function tamperSelfTest() {
  if (!existsSync(abs(ARTIFACTS))) {
    console.error(`[install-matrix] TAMPER=1 requires ${ARTIFACTS}; run scripts/package-artifacts.sh first`);
    process.exit(2);
  }
  const manifest = JSON.parse(readFileSync(abs(ARTIFACTS), 'utf8'));
  const built = (manifest.artifacts || []).filter((a) => a.status === 'built' && a.path && a.sha256);
  const target = built.find((a) => a.kind === 'daemon-bundle') || built[0];
  if (!target) {
    console.error('[install-matrix] TAMPER=1 found no built artifact in the manifest to tamper with');
    process.exit(2);
  }

  const tmp = mkdtempSync(join(tmpdir(), 'faktor-matrix-tamper-'));
  const copy = join(tmp, basename(target.path));
  const bytes = readFileSync(abs(target.path));
  if (bytes.length === 0) {
    console.error('[install-matrix] TAMPER=1 target artifact is empty');
    process.exit(2);
  }
  bytes[Math.floor(bytes.length / 2)] ^= 0xff;
  writeFileSync(copy, bytes);

  const tampered = JSON.parse(JSON.stringify(manifest));
  const entry = tampered.artifacts.find((a) => a.name === target.name);
  entry.path = copy;
  const tamperedManifest = join(tmp, 'artifacts.json');
  writeFileSync(tamperedManifest, `${JSON.stringify(tampered, null, 2)}\n`);
  const tOut = join(tmp, 'out');
  mkdirSync(tOut, { recursive: true });

  const res = spawnSync(process.execPath, [fileURLToPath(import.meta.url)], {
    encoding: 'utf8',
    cwd: ROOT,
    maxBuffer: 64 * 1024 * 1024,
    env: { ...process.env, MATRIX_ARTIFACTS: tamperedManifest, MATRIX_OUT_DIR: tOut, TAMPER: '0' },
  });
  const combined = `${res.stdout || ''}${res.stderr || ''}`;
  const rejected = res.status !== 0;
  const shaEvidence = /sha256/.test(combined) && /recorded=.*actual=/.test(combined);
  let checkEvidence = '';
  try {
    const inner = JSON.parse(readFileSync(join(tOut, 'install-matrix.json'), 'utf8'));
    const failed = (inner.artifacts || []).find((a) => a.name === target.name);
    const shaCheck = failed && failed.checks.find((c) => c.name === 'sha256');
    checkEvidence = shaCheck ? `sha256 check ${shaCheck.status}: ${shaCheck.detail}` : 'no sha256 check recorded';
  } catch {
    checkEvidence = 'inner report unavailable';
  }
  const status = rejected ? 'pass' : 'fail';
  const report = {
    schema: 'faktor-install-matrix-tamper/v1',
    status,
    artifact: { name: target.name, kind: target.kind, original: abs(target.path), tampered_copy: copy },
    expected: 'verifier rejects the byte-flipped copy',
    observed: { exit: res.status, rejected, sha_evidence: shaEvidence, inner_check: checkEvidence },
    timestamp: new Date().toISOString(),
  };
  writeJson(join(OUT_DIR, 'install-matrix-tamper.json'), report);
  rmSync(tmp, { recursive: true, force: true });

  if (rejected) {
    console.log(`[install-matrix] TAMPER self-test PASS: byte-flipped copy of ${target.name} rejected (exit=${res.status})`);
    console.log(`[install-matrix] ${checkEvidence}`);
    process.exit(0);
  }
  console.error(`[install-matrix] TAMPER self-test FAIL: byte-flipped copy of ${target.name} was ACCEPTED (exit=${res.status})`);
  console.error(combined.split('\n').slice(-20).join('\n'));
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Normal run.
// ---------------------------------------------------------------------------
function main() {
  const report = {
    schema: 'faktor-install-matrix/v1',
    status: 'fail',
    commit: null,
    host: {
      os: process.platform,
      arch: process.arch,
      node: process.version,
      tools: {
        tar: run('tar', ['--version']).status === 0,
        unzip: run('unzip', ['-v']).status === 0,
      },
    },
    source: ARTIFACTS,
    timestamp: new Date().toISOString(),
    required: REQUIRE,
    artifacts: [],
    skipped: [],
    residual: RESIDUAL,
    tamper_self_test: null,
  };

  const manifestFile = abs(ARTIFACTS);
  if (!existsSync(manifestFile)) {
    report.skipped.push({ name: 'artifacts-manifest', reason: `${ARTIFACTS} not found; run scripts/package-artifacts.sh first` });
    writeJson(join(OUT_DIR, 'install-matrix.json'), report);
    console.error(`[install-matrix] FAIL: ${ARTIFACTS} not found; run scripts/package-artifacts.sh first`);
    process.exit(1);
  }
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestFile, 'utf8'));
  } catch (e) {
    report.skipped.push({ name: 'artifacts-manifest', reason: `unparseable JSON: ${e.message}` });
    writeJson(join(OUT_DIR, 'install-matrix.json'), report);
    console.error(`[install-matrix] FAIL: ${ARTIFACTS} is not valid JSON: ${e.message}`);
    process.exit(1);
  }
  report.commit = manifest.commit || null;
  report.artifacts_manifest_status = manifest.status || null;

  let failures = 0;
  for (const a of manifest.artifacts || []) {
    if (a.status !== 'built') {
      report.skipped.push({ name: a.name, reason: `${a.status}: ${a.detail || 'not built'}` });
      if (a.status === 'failed') failures += 1;
      continue;
    }
    const result = verifyArtifact(a);
    report.artifacts.push(result);
    if (result.status !== 'pass') failures += 1;
    const failedChecks = result.checks.filter((c) => c.status === 'fail');
    console.log(`[install-matrix] ${result.status.toUpperCase().padEnd(4)} ${result.name} (${result.kind})`);
    for (const c of result.checks) {
      console.log(`[install-matrix]   ${c.status === 'pass' ? 'ok  ' : 'FAIL'} ${c.name}: ${c.detail}`);
    }
    if (failedChecks.length === 0) {
      console.log(`[install-matrix]   evidence: ${JSON.stringify(result.evidence)}`);
    }
  }
  for (const s of manifest.skipped || []) {
    report.skipped.push({ name: s.name, reason: s.reason });
  }

  for (const kind of REQUIRE) {
    const ok = report.artifacts.some((a) => a.kind === kind && a.status === 'pass');
    if (!ok) {
      failures += 1;
      report.skipped.push({ name: `required:${kind}`, reason: `MATRIX_REQUIRE=${REQUIRE.join(':')} but no passing '${kind}' artifact was verified` });
      console.error(`[install-matrix] FAIL required kind '${kind}' did not verify`);
    }
  }

  report.status = failures === 0 ? 'pass' : 'fail';
  writeJson(join(OUT_DIR, 'install-matrix.json'), report);
  const checksTotal = report.artifacts.reduce((n, a) => n + a.checks.length, 0);
  console.log(`[install-matrix] status=${report.status} artifacts=${report.artifacts.length} checks=${checksTotal} skipped=${report.skipped.length} report=${join(OUT_DIR, 'install-matrix.json')}`);
  process.exit(failures === 0 ? 0 : 1);
}

if (TAMPER) {
  tamperSelfTest();
} else {
  main();
}
