# 在 VS Code 中打开 MegaMoE Balanced 源码泳道图

这个可选扩展从图中调用点直接打开旁边的原生源码编辑器，并定位到对应行。使用仓库里的 `index.html`、`source-evidence.json` 与 `source-excerpts.json`，不需要 NPU、CANN、npm 或本地 HTTP 服务。

## 安装与打开

在 `docs/megamoe-balanced-visual` 下执行：

```bash
python tools/package-editor.py
code --install-extension .artifacts/megamoe-balanced-source-guide-0.1.0.vsix
```

然后在 VS Code 打开本源码仓库，运行命令 **MegaMoE Balanced: 打开源码泳道图**。也可直接打开 `docs/megamoe-balanced-visual/index.html`，或者选择 **Reopen Editor With… → MegaMoE Balanced 源码泳道图**。多工作区中发现多份图时会让你选择。

图中源码链接通过 VS Code 消息桥定位，目标固定为导览旁边的编辑列，连续点击会复用该列。普通浏览器仍可阅读同一份离线页面。Remote SSH 中请将扩展安装到远程工作区，源码保持在同一远程仓库中打开。

## 当前源码与参考快照

- `source-evidence.json` 的文件链接位于本仓内：算子代码链接到本仓源码，框架代码链接到 `reference-src/` 的固定参考快照。
- `reference-src/` 是读图用的历史源码快照，版本见来源记录；它不表示当前整网环境正在安装或执行该版本框架。
- 定位使用生成时内嵌的原始调用点及附近源码。只在文件头增加注释等行号移动时，可以重新找到调用点；调用点改名或上下文无法唯一匹配时，应更新导览后重开页面。
- 消息只接受已声明的调用点编号和 `from` / `to` 侧，不接受网页传入任意文件路径。扩展不会修改源码、运行训练或执行页面传入的命令。

## 开发验证

```bash
node tools/editor-check.cjs
python tools/package-editor.py
```

前一项用 VS Code API 替身加载真实扩展，检查当前映射、两侧源码跳转、行号移位、路径限制、远程 URI、CSP 与固定列；它不替代真实编辑器的界面验证。后一项只生成 VSIX，不安装或发布扩展。

## 许可证

此扩展从 DeepEP 的 MegaMoE Train 源码导览扩展改编，许可证和原始版权声明完整保留于本目录的 [LICENSE](LICENSE)，适用于扩展本身。这里不声明或变更整个算子仓库及 `reference-src/` 中第三方框架的许可证；参考快照保留各自原有声明。
