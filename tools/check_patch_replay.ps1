$ErrorActionPreference = 'Stop'
$experimentRoot = 'D:\megamoe_train\moonep-benchmark-alignment-20261112\ep8-three-layer-comparison-20261113'
$entries = @(
    @{ Name='native-balanced'; Sha='48ef24a6' },
    @{ Name='native-shared'; Sha='7626dc9c' },
    @{ Name='FSDPTurbo'; Sha='9c443c61' },
    @{ Name='MindSpeed-LLM'; Sha='dd5ee032' }
)
$records = @()
foreach ($entry in $entries) {
    $target = Join-Path $experimentRoot ('artifacts/replay-check-context-lf/' + $entry.Name)
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $archive = Join-Path $experimentRoot ('artifacts/baselines/' + $entry.Name + '-' + $entry.Sha + '.tar')
    $sourceRepository = Join-Path $experimentRoot ('source/' + $entry.Name)
    $diffPaths = @(git -C $sourceRepository diff --name-only)
    $existing = @()
    foreach ($relative in $diffPaths) {
        git -C $sourceRepository cat-file -e ('HEAD:' + $relative) 2>$null
        if ($LASTEXITCODE -eq 0) { $existing += $relative }
    }
    $tarArguments = @('-xf', $archive, '-C', $target) + $existing
    & tar @tarArguments
    if ($LASTEXITCODE -ne 0) { throw 'Baseline context extraction failed' }
    git -C $target init --quiet
    git -C $target config core.autocrlf false
    $patchFile = Join-Path $experimentRoot ('artifacts/patches/' + $entry.Name + '.patch')
    git -C $target apply --check $patchFile
    if ($LASTEXITCODE -ne 0) { throw ('Patch check failed: ' + $entry.Name) }
    git -C $target apply $patchFile
    if ($LASTEXITCODE -ne 0) { throw ('Patch replay failed: ' + $entry.Name) }
    $diffPaths = git -C (Join-Path $experimentRoot ('source/' + $entry.Name)) diff --name-only
    foreach ($relative in $diffPaths) {
        $expected = Join-Path $experimentRoot ('source/' + $entry.Name + '/' + $relative)
        $actual = Join-Path $target $relative
        if (Test-Path -LiteralPath $expected -PathType Leaf) {
            if ((Get-FileHash -LiteralPath $expected -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $actual -Algorithm SHA256).Hash) { throw ('Replayed content mismatch: ' + $relative) }
        }
    }
    $records += @{ repository=$entry.Name; patch_applies=$true; changed_file_hashes_match=$true; scope="modified baseline contexts"; target=$target }
}
$records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $experimentRoot 'artifacts/patch-replay-validation.json') -Encoding utf8
$records | ConvertTo-Json -Depth 5
