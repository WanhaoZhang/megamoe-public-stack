// Load the real extension with a small VS Code host substitute. No editor/UI is launched.
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL, fileURLToPath } = require('node:url');
const vm = require('node:vm');
const { createRequire } = require('node:module');

const visual = path.resolve(__dirname, '..');
const root = path.resolve(visual, '../..');
const extensionPath = path.join(visual, 'vscode/extension.cjs');
const indexPath = path.join(visual, 'index.html');
const mappingPath = path.join(visual, 'source-evidence.json');
const REMOTE_ROOT = '/workspace/MegaMoE';

class Uri {
  constructor(value) { this.url = new URL(value); }
  static file(value) { return new Uri(pathToFileURL(value)); }
  static remote(value) { return new Uri(`vscode-remote://ssh-remote+source${REMOTE_ROOT}/${path.relative(root, value).split(path.sep).join('/')}`); }
  static joinPath(base, ...parts) {
    const url = new URL(base.url); url.pathname = path.posix.join(base.path, ...parts); return new Uri(url);
  }
  get path() { return decodeURIComponent(this.url.pathname); }
  get scheme() { return this.url.protocol.slice(0, -1); }
  get authority() { return this.url.host; }
  get fsPath() {
    if (this.scheme === 'vscode-remote') {
      assert(this.path.startsWith(REMOTE_ROOT + '/'));
      return path.join(root, this.path.slice(REMOTE_ROOT.length + 1));
    }
    return fileURLToPath(this.url);
  }
  with(change) {
    const uri = new Uri(this.url);
    if ('query' in change) uri.url.search = change.query;
    if ('fragment' in change) uri.url.hash = change.fragment;
    return uri;
  }
}
class Range {
  constructor(line, character, endLine, endCharacter) {
    this.start = { line, character }; this.end = { line: endLine, character: endCharacter };
  }
}

async function harness(remote = false, manifestOverride) {
  let provider, receive, command;
  const opened = [], shown = [], errors = [], revealed = [], executed = [], documents = new Map();
  const indexUri = remote ? Uri.remote(indexPath) : Uri.file(indexPath);
  const host = {
    Uri, Range, ViewColumn: { Beside: -2, Active: -1 }, TextEditorRevealType: { InCenter: 0 },
    workspace: {
      fs: { readFile: uri => manifestOverride && uri.fsPath === mappingPath ? Buffer.from(JSON.stringify(manifestOverride)) : fs.readFile(uri.fsPath) },
      async openTextDocument(uri) {
        opened.push(uri);
        if (!documents.has(uri.fsPath)) {
          documents.set(uri.fsPath, {
            uri, text: await fs.readFile(uri.fsPath, 'utf8'), getText() { return this.text; },
            get lineCount() { return this.text.split(/\r\n|\r|\n/).length; },
          });
        }
        return documents.get(uri.fsPath);
      },
    },
    window: {
      activeTextEditor: { document: { uri: indexUri } },
      registerCustomEditorProvider(type, value) { assert.equal(type, 'megamoeBalanced.sourceGuide'); provider = value; return { dispose() {} }; },
      async showTextDocument(document, options) {
        shown.push({ document, options });
        host.window.activeTextEditor = { document, viewColumn: options.viewColumn };
        return { revealRange(range, type) { revealed.push({ range, type }); } };
      },
      showErrorMessage(message) { errors.push(message); },
    },
    commands: {
      registerCommand(id, value) { assert.equal(id, 'megamoeBalanced.openSourceGuide'); command = value; return { dispose() {} }; },
      async executeCommand(...args) { executed.push(args); },
    },
  };
  const module = { exports: {} }, nativeRequire = createRequire(extensionPath);
  vm.runInNewContext(await fs.readFile(extensionPath, 'utf8'), {
    module, exports: module.exports, Buffer, require: name => name === 'vscode' ? host : nativeRequire(name),
  }, { filename: extensionPath });
  module.exports.activate({ subscriptions: [] });
  assert(provider && command);
  const page = { html: '', options: {}, onDidReceiveMessage(handler) { receive = handler; return { dispose() {} }; } };
  const panel = { viewColumn: 1, webview: page, onDidDispose() {} };
  await provider.resolveCustomEditor(provider.openCustomDocument(indexUri), panel, { isCancellationRequested: false });
  return { host, panel, page, receive, provider, command, opened, shown, errors, revealed, executed, documents, indexUri };
}

async function main() {
  const evidence = JSON.parse(await fs.readFile(mappingPath, 'utf8'));
  const pkg = JSON.parse(await fs.readFile(path.join(visual, 'vscode/package.json'), 'utf8'));
  assert.equal(pkg.name, 'megamoe-balanced-source-guide');
  assert.equal(pkg.version, '0.1.0'); assert.equal(pkg.publisher, 'deepep-local');
  assert.equal(pkg.dependencies, undefined); assert.equal(pkg.contributes.customEditors[0].priority, 'default');
  assert.equal(pkg.contributes.customEditors[0].selector[0].filenamePattern, '**/docs/megamoe-balanced-visual/index.html');
  assert((await fs.readFile(path.join(visual, 'vscode/LICENSE'), 'utf8')).includes('BSD 2-Clause License'));
  const row = evidence.rows.find(value => value.from !== value.to);
  assert(row, 'A source-backed cross-file callpoint is required');
  const filePath = side => path.resolve(visual, decodeURIComponent(evidence.files[row[side]].href));
  const h = await harness();
  const nonce = h.page.html.match(/script-src 'nonce-([^']+)'/)?.[1];
  assert(nonce && h.page.html.includes("default-src 'none'"));
  const tags = [...h.page.html.matchAll(/<script\b[^>]*>/g)];
  assert(tags.length && tags.every(match => match[0].includes(`nonce="${nonce}"`)));
  assert.equal(h.page.options.enableScripts, true); assert.equal(h.page.options.localResourceRoots.length, 0);
  for (const side of ['from', 'to']) {
    await h.receive({ type: 'openSource', callpointId: row.id, side });
    assert.equal(h.opened.at(-1).fsPath, filePath(side));
    assert.equal(h.shown.at(-1).options.selection.start.line, row[`${side}Line`] - 1);
  }
  const fromDocument = h.documents.get(filePath('from'));
  fromDocument.text = '// Unsaved prefix for live anchor lookup\n' + fromDocument.text;
  await h.receive({ type: 'openSource', callpointId: row.id, side: 'from' });
  assert.equal(h.shown.at(-1).options.selection.start.line, row.fromLine);
  await Promise.all([
    h.receive({ type: 'openSource', callpointId: row.id, side: 'from' }),
    h.receive({ type: 'openSource', callpointId: row.id, side: 'to' }),
  ]);
  assert(h.shown.every(value => value.options.viewColumn === 2 && value.options.preview === false));
  assert(h.revealed.every(value => value.type === h.host.TextEditorRevealType.InCenter));
  const openCount = h.opened.length;
  await h.receive({ type: 'openSource', callpointId: '../../outside', side: 'from', path: '../../outside' });
  await h.receive({ type: 'openSource', callpointId: row.id, side: 'constructor' });
  assert.equal(h.opened.length, openCount); assert.equal(h.errors.length, 2);
  await h.receive({ type: 'openMapping', path: '../../outside.json' });
  assert.equal(h.opened.at(-1).fsPath, mappingPath);
  assert.equal(h.shown.at(-1).options.selection.start.line, 0);
  h.host.window.activeTextEditor = { document: { uri: h.indexUri } };
  await h.command();
  assert.equal(h.executed[0][0], 'vscode.openWith');
  assert.equal(h.executed[0][1].fsPath, indexPath); assert.equal(h.executed[0][2], 'megamoeBalanced.sourceGuide');

  const snapshotRow = evidence.rows.find(value => evidence.files[value.to].href.startsWith('reference-src/'));
  assert(snapshotRow, 'Framework reference snapshots must have an explicit local target');
  await h.receive({ type: 'openSource', callpointId: snapshotRow.id, side: 'to' });
  assert(h.opened.at(-1).fsPath.includes(path.join('reference-src', '')));

  const remote = await harness(true);
  await remote.receive({ type: 'openSource', callpointId: row.id, side: 'to' });
  assert.equal(remote.errors.length, 0);
  assert.equal(remote.opened.at(-1).scheme, 'vscode-remote');
  assert.equal(remote.opened.at(-1).authority, 'ssh-remote+source');
  assert.equal(remote.opened.at(-1).fsPath, filePath('to'));
  assert.equal(remote.shown.at(-1).options.viewColumn, 2);

  const poisoned = structuredClone(evidence);
  poisoned.files[row.from].href = '../../../outside.py';
  await assert.rejects(() => harness(false, poisoned), /当前仓库内/);
  const encoded = structuredClone(evidence);
  encoded.files[row.from].href = '%2e%2e/%2e%2e/%2e%2e/outside.py';
  await assert.rejects(() => harness(false, encoded), /当前仓库内/);
  console.log('PASS: actual extension, current evidence, from/to and snapshot sources, live anchor offsets, fixed neighbor column, invalid IDs, path bounds, remote URI and nonce CSP');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
