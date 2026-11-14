// Copyright (c) 2026, DeepEP contributors.
// Adapted from the MegaMoE Train source-guide extension under BSD-2-Clause.
'use strict';

const vscode = require('vscode');
const { randomBytes, createHash } = require('node:crypto');

const VIEW_TYPE = 'megamoeBalanced.sourceGuide';
const COMMAND = 'megamoeBalanced.openSourceGuide';
const INDEX_SUFFIX = '/docs/megamoe-balanced-visual/index.html';
const INDEX_GLOB = '**/docs/megamoe-balanced-visual/index.html';
const decodeText = bytes => Buffer.from(bytes).toString('utf8');
const splitLines = text => text.split(/\r\n|\r|\n/);

function repositoryRoot(uri) {
  if (!uri.path.endsWith(INDEX_SUFFIX)) throw new Error('请打开 docs/megamoe-balanced-visual/index.html。');
  return vscode.Uri.joinPath(uri, '../../..').with({ query: '', fragment: '' });
}

function sourceUri(root, visual, href) {
  if (typeof href !== 'string' || !href || href.startsWith('/') || href.includes('\\') ||
      href.includes('\0') || href.includes('?') || href.includes('#') || /^[A-Za-z][A-Za-z\d+.-]*:/.test(href)) {
    throw new Error('源码映射路径无效，请重新生成导览。');
  }
  const parts = href.split('/').map(part => decodeURIComponent(part));
  if (parts.some(part => !part || /[\\/\0:?#]/.test(part))) throw new Error('源码映射路径无效。');
  const uri = vscode.Uri.joinPath(visual, ...parts);
  if (uri.scheme !== root.scheme || uri.authority !== root.authority ||
      !uri.path.startsWith(root.path.replace(/\/$/, '') + '/')) {
    throw new Error('源码映射必须位于当前仓库内。');
  }
  return uri;
}

function excerptLines(snippets, key, line) {
  const excerpt = snippets[`${key}:${line}`];
  if (typeof excerpt !== 'string') return [];
  return splitLines(excerpt).map(text => {
    const match = text.match(/^\s*(\d+)(?: {2}(.*))?$/);
    return match ? { line: Number(match[1]), text: match[2] || '' } : null;
  }).filter(Boolean);
}

function sourceReferences(manifest, snippets, root, visual) {
  if (!Array.isArray(manifest.rows) || !manifest.files || typeof manifest.files !== 'object') {
    throw new Error('source-evidence.json 内容不完整，请重新生成导览。');
  }
  const references = new Map();
  for (const row of manifest.rows) {
    if (typeof row.id !== 'string' || !row.id || references.has(row.id)) throw new Error('调用点编号无效或重复。');
    const pair = {};
    for (const side of ['from', 'to']) {
      const key = row[side], line = row[`${side}Line`];
      if (typeof key !== 'string' || !Object.hasOwn(manifest.files, key) || !Number.isInteger(line) || line < 1) {
        throw new Error(`调用点 ${row.id} 的源码映射无效。`);
      }
      const file = manifest.files[key];
      if (Number.isInteger(file.lines) && line > file.lines) throw new Error(`调用点 ${row.id} 行号越界。`);
      const context = excerptLines(snippets, key, line);
      const anchor = row[`${side}Anchor`] || context.find(entry => entry.line === line)?.text;
      pair[side] = {
        uri: sourceUri(root, visual, file.href), line, anchor,
        occurrence: row[`${side}Occurrence`], context, sha256: file.sha256,
      };
    }
    references.set(row.id, pair);
  }
  return references;
}

async function sourceLine(document, reference) {
  const lines = splitLines(document.getText());
  let anchor = reference.anchor;
  if (typeof anchor !== 'string' || !anchor.trim()) {
    // Without an embedded anchor, use the recorded line only for the exact
    // source snapshot. Changed content must not silently jump to a stale line.
    const bytes = await vscode.workspace.fs.readFile(reference.uri);
    const hash = createHash('sha256').update(bytes).digest('hex');
    if (hash !== reference.sha256 || document.getText() !== decodeText(bytes)) {
      throw new Error('当前源码已变化且缺少调用点锚点，请更新源码映射。');
    }
    if (reference.line > lines.length) throw new Error('源码行号越界，请更新源码映射。');
    return reference.line - 1;
  }
  const wanted = anchor.trimEnd();
  const matches = [];
  for (let i = 0; i < lines.length; ++i) if (lines[i].trimEnd() === wanted) matches.push(i);
  if (reference.occurrence !== undefined) {
    if (!Number.isInteger(reference.occurrence) || reference.occurrence < 1 || !matches[reference.occurrence - 1] && matches[reference.occurrence - 1] !== 0) {
      throw new Error('源码锚点 occurrence 无效，请更新映射。');
    }
    return matches[reference.occurrence - 1];
  }
  if (matches.length === 1) return matches[0];
  const contextual = matches.map(line => ({
    line,
    score: reference.context.reduce((score, item) => {
      const position = line + item.line - reference.line;
      return score + (item.text.trim() && lines[position]?.trimEnd() === item.text.trimEnd() ? 1 : 0);
    }, 0),
  })).sort((a, b) => b.score - a.score);
  if (contextual[0]?.score >= 2 && contextual[0].score > (contextual[1]?.score || 0)) return contextual[0].line;
  throw new Error('当前源码中找不到唯一调用点，请更新导览后重开。');
}

async function showSource(panel, document, line) {
  const line0 = Math.max(0, Math.min(line, document.lineCount - 1));
  const range = new vscode.Range(line0, 0, line0, 0);
  // Bind the target to the guide's column, not the last focused source editor.
  // Two queued clicks therefore reuse the same neighbor instead of a third group.
  const column = panel.viewColumn;
  const viewColumn = Number.isInteger(column) && column >= 1 ? Math.min(column + 1, 9) : vscode.ViewColumn.Beside;
  const editor = await vscode.window.showTextDocument(document, { viewColumn, selection: range, preview: false });
  editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
}

function readerHtml(html) {
  const nonce = randomBytes(18).toString('base64');
  const policy = `default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';`;
  if (!/<head\b[^>]*>/i.test(html)) throw new Error('导览页面不完整，请重新生成 index.html。');
  return html.replace(/<head\b[^>]*>/i, head => `${head}\n<meta http-equiv="Content-Security-Policy" content="${policy}">`)
    .replace(/<script\b/g, () => `<script nonce="${nonce}"`);
}

async function openGuide() {
  const active = vscode.window.activeTextEditor?.document.uri || vscode.window.tabGroups.activeTabGroup.activeTab?.input?.uri;
  let candidates;
  if (active?.path.endsWith(INDEX_SUFFIX)) candidates = [active];
  else {
    const folder = active && vscode.workspace.getWorkspaceFolder(active);
    if (folder) candidates = await vscode.workspace.findFiles(new vscode.RelativePattern(folder, INDEX_GLOB), '**/node_modules/**');
    if (!candidates?.length) candidates = await vscode.workspace.findFiles(INDEX_GLOB, '**/node_modules/**');
  }
  if (!candidates.length) throw new Error('请先打开包含 docs/megamoe-balanced-visual 的源码仓库。');
  let uri = candidates[0];
  if (candidates.length > 1) {
    const selected = await vscode.window.showQuickPick(candidates.map(candidate => ({
      label: vscode.workspace.asRelativePath(candidate, true), description: candidate.path, uri: candidate,
    })), { placeHolder: '选择 MegaMoE Balanced 泳道图所在仓库' });
    if (!selected) return;
    uri = selected.uri;
  }
  await vscode.commands.executeCommand('vscode.openWith', uri, VIEW_TYPE, { viewColumn: vscode.ViewColumn.Active, preview: false });
}

function activate(context) {
  const reportError = error => vscode.window.showErrorMessage(`MegaMoE Balanced：${error.message || error}`);
  const provider = {
    openCustomDocument(uri) { repositoryRoot(uri); return { uri, dispose() {} }; },
    async resolveCustomEditor(document, panel, token) {
      const root = repositoryRoot(document.uri), visual = vscode.Uri.joinPath(document.uri, '..');
      const mappingUri = vscode.Uri.joinPath(visual, 'source-evidence.json');
      const snippetsUri = vscode.Uri.joinPath(visual, 'source-excerpts.json');
      const [pageBytes, mapBytes, snippetBytes] = await Promise.all([
        vscode.workspace.fs.readFile(document.uri), vscode.workspace.fs.readFile(mappingUri), vscode.workspace.fs.readFile(snippetsUri),
      ]);
      if (token.isCancellationRequested) return;
      const references = sourceReferences(JSON.parse(decodeText(mapBytes)), JSON.parse(decodeText(snippetBytes)), root, visual);
      panel.webview.options = { enableScripts: true, localResourceRoots: [] };
      let pending = Promise.resolve(), disposed = false;
      const listener = panel.webview.onDidReceiveMessage(message => {
        pending = pending.then(async () => {
          if (disposed) return;
          if (message?.type === 'openMapping') {
            await showSource(panel, await vscode.workspace.openTextDocument(mappingUri), 0);
          } else if (message?.type === 'openSource') {
            if (typeof message.callpointId !== 'string' || !['from', 'to'].includes(message.side)) throw new Error('调用点选择无效。');
            const source = references.get(message.callpointId)?.[message.side];
            if (!source) throw new Error('找不到对应调用点，请更新导览。');
            const sourceDocument = await vscode.workspace.openTextDocument(source.uri);
            await showSource(panel, sourceDocument, await sourceLine(sourceDocument, source));
          }
        }).catch(reportError);
        return pending;
      });
      panel.onDidDispose(() => { disposed = true; listener.dispose(); });
      panel.webview.html = readerHtml(decodeText(pageBytes));
    },
  };
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW_TYPE, provider, { webviewOptions: { retainContextWhenHidden: true }, supportsMultipleEditorsPerDocument: false }),
    vscode.commands.registerCommand(COMMAND, () => openGuide().catch(reportError)),
  );
}

module.exports = { activate };
